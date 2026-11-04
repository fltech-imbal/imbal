"""Small neural networks and exact Gaussian KDEs with bounded workspaces."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


DENSITY_EPSILON = 1e-7
CORRELATION_EPSILON = 1e-7


def predict_in_batches(model, features, batch_size=128):
    """Evaluate all rows without passing the entire dataset through the model."""

    if batch_size < 1:
        raise ValueError("Prediction batch size must be positive.")

    # Only the small output arrays survive each iteration. In particular, the
    # KDE's intermediate query-by-reference matrices are not accumulated here.
    prediction_batches = []

    for start_index in range(0, len(features), batch_size):
        feature_batch = features[start_index:start_index + batch_size]
        batch_predictions = model(feature_batch, training=False)
        prediction_batches.append(np.asarray(batch_predictions))

    if not prediction_batches:
        return np.empty((0, int(model.output_shape[-1])), dtype="float32")

    return np.concatenate(prediction_batches, axis=0)


def gaussian_kde_density(
    query_values,
    reference_targets,
    bandwidth,
    query_chunk_size=128,
    reference_chunk_size=4096,
):
    """Evaluate the original Gaussian KDE exactly, without an all-pairs array."""

    queries = np.asarray(query_values, dtype="float64").reshape(-1)
    references = np.asarray(reference_targets, dtype="float64").reshape(-1)

    if not len(references) or bandwidth <= 0:
        raise ValueError("KDE requires reference targets and a positive bandwidth.")

    if query_chunk_size < 1 or reference_chunk_size < 1:
        raise ValueError("KDE chunk sizes must be positive.")

    densities = np.empty(len(queries), dtype="float64")
    normalizer = len(references) * bandwidth * np.sqrt(2.0 * np.pi)

    for query_start in range(0, len(queries), query_chunk_size):
        query_batch = queries[query_start:query_start + query_chunk_size]
        kernel_totals = np.zeros(len(query_batch), dtype="float64")

        # Visit every reference target. Chunking changes workspace size, not
        # bandwidth, sample membership, or the mathematical density estimator.
        for reference_start in range(0, len(references), reference_chunk_size):
            reference_batch = references[
                reference_start:reference_start + reference_chunk_size
            ]

            kernel_values = query_batch[:, None] - reference_batch[None, :]
            kernel_values /= bandwidth

            # Reuse the NumPy buffer for each operation instead of retaining
            # separate difference, squared-distance, and exponential arrays.
            np.square(kernel_values, out=kernel_values)
            kernel_values *= -0.5
            np.exp(kernel_values, out=kernel_values)

            kernel_totals += np.sum(kernel_values, axis=1)

        densities[query_start:query_start + len(query_batch)] = (
            kernel_totals / normalizer
        )

    return densities


def build_density_profile(targets, settings):
    """Keep the October bandwidth and normalization rules unchanged."""

    reference_targets = np.asarray(targets, dtype="float32").reshape(-1)

    if not len(reference_targets):
        raise ValueError("Cannot build a density profile for an empty expert region.")

    # The earlier implementation promotes to float64 before estimating the
    # bandwidth. Preserve that detail rather than altering the density curve.
    bandwidth_targets = reference_targets.astype("float64")
    bandwidth = 0.05

    if len(reference_targets) > 1:
        estimated_bandwidth = (
            1.06
            * np.std(bandwidth_targets, ddof=1)
            * len(reference_targets) ** (-1.0 / 5.0)
        )

        if np.isfinite(estimated_bandwidth):
            bandwidth = max(float(estimated_bandwidth), 0.05)

    reference_densities = gaussian_kde_density(
        reference_targets,
        reference_targets,
        bandwidth,
        settings.density_query_chunk_size,
        settings.density_reference_chunk_size,
    )

    maximum_density = float(np.max(reference_densities))

    if not np.isfinite(maximum_density) or maximum_density <= 0:
        raise ValueError("The density normalization must be positive and finite.")

    return {
        "training_targets": reference_targets,
        "bandwidth": bandwidth,
        "maximum_density": maximum_density,
    }


@keras.utils.register_keras_serializable(package="NovemberSEP")
class NormalizedGaussianKDE(layers.Layer):
    """Serialized KDE that bounds query and reference blocks inside the layer.

    Even model(all_rows) remains bounded during forward evaluation. The nested
    loops run sequentially; only reduced densities survive each query block.
    """

    def __init__(
        self,
        training_targets,
        bandwidth,
        maximum_density,
        query_chunk_size=128,
        reference_chunk_size=4096,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.training_targets_list = np.asarray(
            training_targets, dtype="float32"
        ).reshape(-1).tolist()

        self.bandwidth = float(bandwidth)
        self.maximum_density = float(maximum_density)
        self.query_chunk_size = int(query_chunk_size)
        self.reference_chunk_size = int(reference_chunk_size)

        if not self.training_targets_list or self.bandwidth <= 0:
            raise ValueError("KDE requires reference targets and a positive bandwidth.")

        if self.maximum_density <= 0 or min(
            self.query_chunk_size, self.reference_chunk_size
        ) < 1:
            raise ValueError("Density normalization and chunk sizes must be positive.")

    def call(self, predictions):
        queries = tf.reshape(predictions, [-1])
        references = tf.constant(self.training_targets_list, dtype=predictions.dtype)
        query_count = tf.shape(queries)[0]
        reference_count = len(self.training_targets_list)

        # A TensorArray stores only one density per query. It never stores the
        # large kernel blocks, and supports a dynamic final batch under tf.function.
        density_blocks = tf.TensorArray(
            dtype=predictions.dtype,
            size=0,
            dynamic_size=True,
            infer_shape=False,
            element_shape=tf.TensorShape([None]),
        )

        def evaluate_query_block(query_start, block_index, output_blocks):
            query_batch = queries[query_start:query_start + self.query_chunk_size]
            initial_totals = tf.zeros_like(query_batch)

            def accumulate_reference_block(reference_start, kernel_totals):
                reference_batch = references[
                    reference_start:reference_start + self.reference_chunk_size
                ]

                differences = (
                    query_batch[:, None] - reference_batch[None, :]
                ) / tf.cast(self.bandwidth, predictions.dtype)

                block_totals = tf.reduce_sum(
                    tf.exp(-0.5 * tf.square(differences)), axis=1
                )

                return (
                    reference_start + self.reference_chunk_size,
                    kernel_totals + block_totals,
                )

            # Only one reference block is active at a time. Keeping parallel
            # iterations at one prevents graph execution from multiplying memory.
            _, kernel_totals = tf.while_loop(
                lambda reference_start, totals: reference_start < reference_count,
                accumulate_reference_block,
                (tf.constant(0), initial_totals),
                parallel_iterations=1,
            )

            normalizer = tf.cast(
                reference_count * self.bandwidth * np.sqrt(2.0 * np.pi),
                predictions.dtype,
            )

            relative_density = (kernel_totals / normalizer) / tf.cast(
                self.maximum_density, predictions.dtype
            )

            bounded_density = tf.clip_by_value(
                relative_density, DENSITY_EPSILON, 1.0
            )

            return (
                query_start + self.query_chunk_size,
                block_index + 1,
                output_blocks.write(block_index, bounded_density),
            )

        _, _, completed_blocks = tf.while_loop(
            lambda query_start, block_index, blocks: query_start < query_count,
            evaluate_query_block,
            (tf.constant(0), tf.constant(0), density_blocks),
            parallel_iterations=1,
        )

        # Explicitly handle empty inputs without asking an empty TensorArray to
        # infer its concatenated shape. Normal nonempty calls return [rows, 1].
        flat_densities = tf.cond(
            query_count > 0,
            lambda: completed_blocks.concat(),
            lambda: tf.zeros([0], dtype=predictions.dtype),
        )

        return tf.reshape(flat_densities, [-1, 1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def get_config(self):
        return {
            **super().get_config(),
            "training_targets": self.training_targets_list,
            "bandwidth": self.bandwidth,
            "maximum_density": self.maximum_density,
            "query_chunk_size": self.query_chunk_size,
            "reference_chunk_size": self.reference_chunk_size,
        }


@keras.utils.register_keras_serializable(package="NovemberSEP")
class NormalizeExpertWeights(layers.Layer):
    """Normalize the two trust values, retaining the original epsilon guard."""

    def call(self, values):
        weight_totals = tf.reduce_sum(values, axis=1, keepdims=True)
        return values / tf.maximum(weight_totals, DENSITY_EPSILON)


@keras.utils.register_keras_serializable(package="NovemberSEP")
class ExpertMSEPCCLoss(keras.losses.Loss):
    """The original unweighted batch MSE + PCC loss for classification gates."""

    def __init__(self, pcc_weight=2.0, name="expert_mse_pcc_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pcc_weight = float(pcc_weight)

    def call(self, y_true, y_pred):
        targets = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        predictions = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        mean_squared_error = tf.reduce_mean(tf.square(targets - predictions))

        # Preserve the unweighted implementation's sums and epsilon checks.
        # Replacing these with weighted variances would change tiny-variance cases.
        centered_targets = targets - tf.reduce_mean(targets)
        centered_predictions = predictions - tf.reduce_mean(predictions)

        target_squared_sum = tf.reduce_sum(tf.square(centered_targets))
        prediction_squared_sum = tf.reduce_sum(tf.square(centered_predictions))
        covariance_sum = tf.reduce_sum(centered_targets * centered_predictions)

        denominator = tf.sqrt(tf.maximum(
            target_squared_sum * prediction_squared_sum, CORRELATION_EPSILON
        ))

        correlation = tf.clip_by_value(covariance_sum / denominator, -1.0, 1.0)
        valid_correlation = tf.logical_and(
            target_squared_sum > CORRELATION_EPSILON,
            prediction_squared_sum > CORRELATION_EPSILON,
        )

        correlation_penalty = tf.where(valid_correlation, 1.0 - correlation, 0.0)
        return mean_squared_error + self.pcc_weight * correlation_penalty

    def get_config(self):
        return {**super().get_config(), "pcc_weight": self.pcc_weight}


@keras.utils.register_keras_serializable(package="NovemberSEP")
class WeightedExpertMSEPCCLoss(ExpertMSEPCCLoss):
    """Apply packed reciprocal weights inside both terms of the expert loss."""

    def call(self, y_true, y_pred):
        packed_targets = tf.cast(y_true, tf.float32)
        targets = tf.reshape(packed_targets[:, 0], [-1])
        weights = tf.maximum(tf.reshape(packed_targets[:, 1], [-1]), 0.0)
        predictions = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

        # Normalizing within the loss matches October's weighted MSE/PCC rule.
        # Passing these weights to model.fit would not weight a scalar PCC correctly.
        total_weight = tf.maximum(tf.reduce_sum(weights), CORRELATION_EPSILON)
        normalized_weights = weights / total_weight

        mean_squared_error = tf.reduce_sum(
            normalized_weights * tf.square(targets - predictions)
        )

        target_mean = tf.reduce_sum(normalized_weights * targets)
        prediction_mean = tf.reduce_sum(normalized_weights * predictions)
        centered_targets = targets - target_mean
        centered_predictions = predictions - prediction_mean

        covariance = tf.reduce_sum(
            normalized_weights * centered_targets * centered_predictions
        )

        target_variance = tf.reduce_sum(normalized_weights * tf.square(centered_targets))
        prediction_variance = tf.reduce_sum(
            normalized_weights * tf.square(centered_predictions)
        )

        denominator = tf.sqrt(tf.maximum(
            target_variance * prediction_variance, CORRELATION_EPSILON
        ))

        correlation = tf.clip_by_value(covariance / denominator, -1.0, 1.0)
        valid_correlation = tf.logical_and(
            target_variance > CORRELATION_EPSILON,
            prediction_variance > CORRELATION_EPSILON,
        )

        correlation_penalty = tf.where(valid_correlation, 1.0 - correlation, 0.0)
        return mean_squared_error + self.pcc_weight * correlation_penalty


def build_expert(expert_name, input_count, settings):
    inputs = keras.Input(shape=(input_count,), name=f"{expert_name}_input")
    hidden = inputs

    # Keep layer order and names compatible with the October saved networks.
    # Their small 18/12/8/6 architecture is not the source of the memory spike.
    for layer_number, units in enumerate((18, 12, 8), start=1):
        hidden = layers.Dense(
            units, activation="relu", name=f"{expert_name}_dense_{layer_number}"
        )(hidden)

    representation = layers.Dense(
        6, activation="relu", name=f"{expert_name}_representation"
    )(hidden)

    prediction = layers.Dense(1, name=f"{expert_name}_prediction")(representation)
    expert = keras.Model(inputs, prediction, name=expert_name)

    loss = "mse"

    if settings.expert_correlation_loss:
        loss_class = (
            WeightedExpertMSEPCCLoss if settings.regression_gate else ExpertMSEPCCLoss
        )
        loss = loss_class(pcc_weight=settings.correlation_weight)

    expert.compile(
        optimizer=keras.optimizers.Adam(settings.learning_rate),
        loss=loss,
        metrics=[] if settings.regression_gate else ["mae"],
    )

    return expert


def build_gate(input_count, settings):
    inputs = keras.Input(shape=(input_count,), name="gate_input")
    hidden = layers.Dense(18, activation="relu", name="gate_dense_1")(inputs)
    hidden = layers.Dense(12, activation="relu", name="gate_dense_2")(hidden)
    representation = layers.Dense(8, activation="relu", name="gate_representation")(hidden)

    # Independent sigmoid trusts are normalized at the mixing stage. Softmax
    # already sums to one and keeps the original gate's layer names and topology.
    if settings.sigmoid_gate:
        linear_scores = layers.Dense(2, name="gate_linear_outputs")(representation)
        trust = layers.Activation("sigmoid", name="gate_trust")(linear_scores)
    else:
        output_name = "gate_trust" if settings.regression_gate else "gate_probabilities"
        trust = layers.Dense(2, activation="softmax", name=output_name)(representation)

    gate = keras.Model(inputs, trust, name="gate")

    if not settings.regression_gate:
        gate.compile(
            optimizer=keras.optimizers.Adam(settings.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    return gate


def build_ensemble(common_expert, rare_expert, gate, density_profiles, settings):
    """Build the exportable inference model, with protected KDE layers."""

    common_expert.trainable = False
    rare_expert.trainable = False
    gate.trainable = False

    inputs = keras.Input(shape=common_expert.input_shape[1:], name="features")
    common_prediction = common_expert(inputs, training=False)
    rare_prediction = rare_expert(inputs, training=False)
    gate_trust = gate(inputs, training=False)

    expert_predictions = layers.Concatenate(name="expert_predictions")(
        [common_prediction, rare_prediction]
    )

    if settings.density_at_inference:
        common_quality = NormalizedGaussianKDE(
            **density_profiles[0],
            query_chunk_size=settings.density_query_chunk_size,
            reference_chunk_size=settings.density_reference_chunk_size,
            name="common_expert_density",
        )(common_prediction)

        rare_quality = NormalizedGaussianKDE(
            **density_profiles[1],
            query_chunk_size=settings.density_query_chunk_size,
            reference_chunk_size=settings.density_reference_chunk_size,
            name="rare_expert_density",
        )(rare_prediction)

        quality = layers.Concatenate(name="expert_densities")(
            [common_quality, rare_quality]
        )

        weighted_trust = layers.Multiply(name="density_weighted_gate")([gate_trust, quality])
        mixture_weights = NormalizeExpertWeights(name="normalized_expert_weights")(
            weighted_trust
        )
    elif settings.sigmoid_gate:
        # Independent sigmoid outputs need not sum to one. Normalize them even
        # without density, matching the gate trainer and cached metric calculation.
        mixture_weights = NormalizeExpertWeights(name="normalized_gate_trust")(
            gate_trust
        )
    else:
        mixture_weights = gate_trust

    prediction = layers.Dot(axes=1, name="gated_prediction")(
        [expert_predictions, mixture_weights]
    )

    return keras.Model(inputs, prediction, name="two_expert_gated_ensemble")


def build_cached_gate_trainer(gate, input_count, settings):
    """Train only the gate, using precomputed predictions and density qualities."""

    gate.trainable = True
    features = keras.Input(shape=(input_count,), name="features")
    expert_predictions = keras.Input(shape=(2,), name="cached_predictions")
    gate_trust = gate(features)

    # These extra inputs are constants from the fixed experts, not learned
    # features for the gate. The gate still sees exactly the original features.
    model_inputs = {"features": features, "cached_predictions": expert_predictions}

    if settings.density_during_gate_training:
        quality = keras.Input(shape=(2,), name="cached_quality")
        model_inputs["cached_quality"] = quality
        weighted_trust = layers.Multiply()([gate_trust, quality])
        mixture_weights = NormalizeExpertWeights()(weighted_trust)
    elif settings.sigmoid_gate:
        mixture_weights = NormalizeExpertWeights()(gate_trust)
    else:
        mixture_weights = gate_trust

    prediction = layers.Dot(axes=1)([expert_predictions, mixture_weights])
    trainer = keras.Model(model_inputs, prediction, name="cached_gate_trainer")

    trainer.compile(
        optimizer=keras.optimizers.Adam(settings.learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return trainer

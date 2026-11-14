"""Run this file in PyCharm to compare the two SEP-C gates.

Keep sep_c_inputs.zip beside this file. The split is read directly from the
archive, so the script can be run from any working directory.
Each training run saves the shared experts first, then both gates, their assembled
ensembles and held-out metrics under runs/<timestamp>/ beside this script.
LOAD_SAVED_MODEL selects saved-model evaluation or fresh training.
Set SHOW_VISUALIZATIONS to display and save a separate imbal test prediction
plot for each gate in the selected or newly trained run folder.
The console compares both gates on the five worst rare regression errors.
"""

from dataclasses import dataclass, replace
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import random
from zipfile import ZipFile

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KernelDensity
from tensorflow import keras
from tensorflow.keras import layers

HERE = Path(__file__).resolve().parent
LOAD_SAVED_MODEL = True  # False trains a new pair and saves it in a new run folder.
SHOW_VISUALIZATIONS = False
# None loads the newest timestamped run containing both ensembles. To choose a
# specific run, use e.g. HERE / "runs" / "20261111-214558-817660".
SAVED_MODEL_DIRECTORY = None
DATA_ARCHIVE = HERE / "sep_c_inputs.zip"
TARGET = "ln_peak_intensity"
RARE_THRESHOLD = np.log(10)
DENSITY_EPSILON = 1e-7
CORRELATION_EPSILON = 1e-7


@dataclass(frozen=True)
class ExperimentSettings:
    run_name: str
    regression_gate: bool = False
    sigmoid_gate: bool = False
    density_at_inference: bool = True
    density_during_gate_training: bool = False
    seed: int = 42
    fold_count: int = 5
    maximum_epochs: int = 500
    early_stopping_patience: int = 100
    training_batch_size: int = 32
    learning_rate: float = 0.001
    correlation_weight: float = 2.0
    alpha_candidates: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)
    rare_class_weight_candidates: tuple = tuple(range(1, 11))
    prediction_batch_size: int = 128
    density_query_chunk_size: int = 128
    density_reference_chunk_size: int = 4096


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


@keras.utils.register_keras_serializable(package="MatchedSEPC")
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


@keras.utils.register_keras_serializable(package="MatchedSEPC")
class NormalizeExpertWeights(layers.Layer):
    """Normalize the two trust values, retaining the original epsilon guard."""

    def call(self, values):
        weight_totals = tf.reduce_sum(values, axis=1, keepdims=True)
        return values / tf.maximum(weight_totals, DENSITY_EPSILON)


@keras.utils.register_keras_serializable(package="MatchedSEPC")
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

    expert.compile(
        optimizer=keras.optimizers.Adam(settings.learning_rate),
        loss=ExpertMSEPCCLoss(pcc_weight=settings.correlation_weight),
        metrics=["mae"],
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


def combine_expert_outputs(
    expert_predictions,
    gate_trust,
    density_quality=None,
    normalize_trust=False,
):
    """Apply the same mixing layers during training, evaluation, and export.

    Every input has columns [common, rare]. Inputs can be symbolic Keras
    tensors when building a model, or numerical arrays for cached evaluation.
    These layers have no learned parameters; only the gate learns how to mix.
    """

    # Quality is prepared by the caller: cached during training or calculated
    # by bounded KDE layers for new inputs. None explicitly disables quality.
    if density_quality is not None:
        weighted_trust = layers.Multiply(name="density_weighted_gate")(
            [gate_trust, density_quality]
        )
        mixture_weights = NormalizeExpertWeights(name="normalized_expert_weights")(
            weighted_trust
        )
    elif normalize_trust:
        # Independent sigmoid outputs need normalization even without quality.
        # Keep the existing epsilon safeguard and saved layer names unchanged.
        mixture_weights = NormalizeExpertWeights(name="normalized_gate_trust")(
            gate_trust
        )
    else:
        # Softmax outputs already sum to one. Preserve that original path.
        mixture_weights = gate_trust

    # Reduce only across the two experts within each sample. Return the weights
    # as well so diagnostics can inspect the exact mixture used for prediction.
    prediction = layers.Dot(axes=1, name="gated_prediction")(
        [expert_predictions, mixture_weights]
    )

    return prediction, mixture_weights


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

    # Inference calculates qualities from the current expert predictions.
    # Passing None makes the quality-free experiment explicit at this boundary.
    quality = None

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

    prediction, _ = combine_expert_outputs(
        expert_predictions,
        gate_trust,
        density_quality=quality,
        normalize_trust=settings.sigmoid_gate,
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

    # Training reuses fixed qualities instead of recalculating their KDE.
    # Inference-only-quality experiments deliberately pass None at this stage.
    quality = None

    if settings.density_during_gate_training:
        quality = keras.Input(shape=(2,), name="cached_quality")
        model_inputs["cached_quality"] = quality

    prediction, _ = combine_expert_outputs(
        expert_predictions,
        gate_trust,
        density_quality=quality,
        normalize_trust=settings.sigmoid_gate,
    )

    trainer = keras.Model(model_inputs, prediction, name="cached_gate_trainer")

    trainer.compile(
        optimizer=keras.optimizers.Adam(settings.learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return trainer


def set_global_determinism(seed):
    """Retain the October seeds for initialization and fold construction."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


@dataclass
class ExpertPredictionCache:
    """Four float32 values per row, rather than a row-by-reference matrix."""

    predictions: np.ndarray
    density_quality: np.ndarray

    def training_inputs(self, features, indices, settings):
        inputs = {
            "features": features[indices],
            "cached_predictions": self.predictions[indices],
        }

        if settings.density_during_gate_training:
            inputs["cached_quality"] = self.density_quality[indices]

        return inputs


def create_prediction_cache(common_expert, rare_expert, features, profiles, settings):
    """Evaluate the fixed experts once; share results across folds and reports."""

    prediction_columns = []
    quality_columns = []

    for expert_index, expert in enumerate((common_expert, rare_expert)):
        predictions = predict_in_batches(
            expert, features, settings.prediction_batch_size
        )

        prediction_columns.append(predictions.reshape(-1))

        layer = NormalizedGaussianKDE(
            **profiles[expert_index], query_chunk_size=settings.density_query_chunk_size,
            reference_chunk_size=settings.density_reference_chunk_size,
        )
        relative_density = np.asarray(layer(predictions)).reshape(-1)
        quality_columns.append(relative_density)

    return ExpertPredictionCache(
        predictions=np.column_stack(prediction_columns),
        density_quality=np.column_stack(quality_columns),
    )


def mix_cached_predictions(gate_trust, cache, use_density, normalize_trust=False):
    """Evaluate the shared mixing layers using the already computed expert outputs."""

    # The caller chooses the experiment's quality policy. The common mixing
    # function then performs exactly the operations used by both Keras models.
    quality = cache.density_quality if use_density else None
    predictions, mixture_weights = combine_expert_outputs(
        cache.predictions,
        gate_trust,
        density_quality=quality,
        normalize_trust=normalize_trust,
    )

    # Metrics consume NumPy arrays. Keep their existing one-value-per-row API
    # while the Keras prediction tensor retains its [batch, 1] output shape.
    return np.asarray(predictions).reshape(-1), np.asarray(mixture_weights)


def aore_components(targets, predictions, rare_mask):
    target_values = np.asarray(targets).reshape(-1)
    prediction_values = np.asarray(predictions).reshape(-1)
    rare_mask = np.asarray(rare_mask).reshape(-1).astype(bool)

    if not np.any(rare_mask):
        raise ValueError("AORE requires at least one rare evaluation sample.")

    # AORE averages overall MAE and rare-region MAE, not common and rare MAE.
    # This distinction is retained in every search callback and final report.
    overall_error = mean_absolute_error(target_values, prediction_values)
    rare_error = mean_absolute_error(
        target_values[rare_mask], prediction_values[rare_mask]
    )

    return float(overall_error), float(rare_error), float((overall_error + rare_error) / 2)


class GateAOREEarlyStopping(keras.callbacks.Callback):
    """Measure the same validation mixture without rerunning frozen experts."""

    def __init__(self, gate, features, targets, region_labels, cache, settings):
        super().__init__()
        self.gate = gate
        self.features = features
        self.targets = targets
        self.region_labels = region_labels
        self.cache = cache
        self.settings = settings

        self.best_aore = np.inf
        self.best_epoch = 1
        self.best_weights = None
        self.epochs_without_improvement = 0

    def on_epoch_end(self, epoch, logs=None):
        trust = predict_in_batches(
            self.gate, self.features, self.settings.prediction_batch_size
        )

        # Regression validation follows the training mixture; classification
        # validation follows the inference mixture. Only the explicitly named
        # inference-only-quality variants intentionally differ between stages.
        use_density = (
            self.settings.density_during_gate_training
            if self.settings.regression_gate
            else self.settings.density_at_inference
        )

        predictions, _ = mix_cached_predictions(
            trust,
            self.cache,
            use_density=use_density,
            normalize_trust=self.settings.sigmoid_gate,
        )

        _, _, validation_aore = aore_components(
            self.targets, predictions, self.region_labels
        )

        if logs is not None:
            logs["val_ensemble_aore"] = validation_aore

        if validation_aore < self.best_aore:
            self.best_aore = validation_aore
            self.best_epoch = epoch + 1
            self.best_weights = self.gate.get_weights()
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.settings.early_stopping_patience:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.gate.set_weights(self.best_weights)


def train_classification_experiment_expert(features, targets, expert_name, settings):
    """Select expert epochs by validation loss, then refit on the full region."""

    if len(features) < settings.fold_count:
        raise ValueError(f"{expert_name} has too few samples for the requested folds.")

    splitter = KFold(
        n_splits=settings.fold_count, shuffle=True, random_state=settings.seed
    )
    selected_epochs = []

    for fold_number, (training_indices, validation_indices) in enumerate(
        splitter.split(features), start=1
    ):
        keras.backend.clear_session()
        set_global_determinism(settings.seed + fold_number)
        expert = build_expert(expert_name, features.shape[1], settings)

        stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=settings.early_stopping_patience,
            mode="min",
            restore_best_weights=True,
        )

        print(f"{expert_name}: epoch-selection fold {fold_number}/{settings.fold_count}")
        history = expert.fit(
            features[training_indices],
            targets[training_indices],
            validation_data=(features[validation_indices], targets[validation_indices]),
            epochs=settings.maximum_epochs,
            batch_size=settings.training_batch_size,
            callbacks=[stopping],
            shuffle=False,
            verbose=0,
        )

        selected_epochs.append(int(np.argmin(history.history["val_loss"]) + 1))

        # Histories and callbacks reference their models. Release those references
        # before the next clear_session rather than carrying an old fold forward.
        del history, stopping, expert
        gc.collect()

    epoch_count = max(1, int(np.round(np.mean(selected_epochs))))
    expert_seed_offset = 1000 if expert_name == "common_expert" else 2000

    keras.backend.clear_session()
    set_global_determinism(settings.seed + expert_seed_offset)
    expert = build_expert(expert_name, features.shape[1], settings)

    expert.fit(
        features,
        targets,
        epochs=epoch_count,
        batch_size=settings.training_batch_size,
        shuffle=False,
        verbose=0,
    )

    print(f"{expert_name}: final training used {epoch_count} epochs")
    return expert


def train_experts(features, targets, region_labels, settings):
    # Experts still specialize on disjoint target regions. No new samples or
    # test labels enter either the common or rare expert's training set.
    common_mask = np.asarray(region_labels).reshape(-1) == 0
    rare_mask = ~common_mask

    common_expert = train_classification_experiment_expert(
        features[common_mask], targets[common_mask], "common_expert", settings
    )
    rare_expert = train_classification_experiment_expert(
        features[rare_mask], targets[rare_mask], "rare_expert", settings
    )

    common_expert.trainable = False
    rare_expert.trainable = False
    return common_expert, rare_expert


def importance_bandwidth(values):
    """SEP-C's one-dimensional histogram/KL bandwidth search for gate weights."""
    bins = int(np.ceil(len(values) / 100))
    lower = values.min()
    upper = values.max() + np.float32(1e-6)
    padding = (upper - lower) * 0.01
    lower, upper = lower - padding, upper + padding
    step = (upper - lower) / bins
    indices = np.clip(np.floor((values - lower) / step).astype(int), 0, bins - 1)
    histogram = np.bincount(indices, minlength=bins) / step / len(values)

    # Keep the original ten samples per histogram bin and its half-bin offset.
    starts = lower + step * np.arange(bins)
    queries = np.concatenate([
        np.linspace(start, start + step, 11)[:-1] + step / 2 for start in starts
    ]).reshape(-1, 1)
    samples = values.reshape(-1, 1)
    best_width = float(np.std(values)) * 1.49
    search_min, search_max = 0.0, best_width * 2
    best_score = np.inf
    while best_score > 0 and search_max - search_min > 1e-9:
        stride = (search_max - search_min) / 10
        candidates = []
        for i in range(10):
            width = stride * (i + 0.5)
            if width > 1e-9:
                kde = KernelDensity(bandwidth=width).fit(samples)
                masses = np.exp(kde.score_samples(queries)).reshape(bins, 10).sum(axis=1)
                masses /= masses.sum()
                score = np.sum(histogram * np.log(histogram / (masses + 1e-6) + 1e-6))
                candidates.append((score, width))
        if not candidates:
            break
        score, width = min(candidates)
        if score > best_score:
            break
        best_score, best_width = score, width
        # Preserve the SEP-C search's sequential updates to the bounds.
        search_min = best_width - stride
        search_max = best_width + (search_max - search_min) / 10
    return best_width


def compute_sample_densities(targets):
    """Evaluate the training-target KDE used for reciprocal importance."""
    values = np.asarray(targets, dtype="float32").ravel()
    if np.ptp(values) == 0:
        return np.ones(len(values))
    width = importance_bandwidth(values)
    order = np.argsort(values)
    sorted_values = values[order]
    span = np.ptp(sorted_values)
    # Same half-open support and sorted summation as SEP-C's atol=0 estimator.
    starts = np.searchsorted(sorted_values, sorted_values - span)
    stops = np.searchsorted(sorted_values, sorted_values + span)
    densities = np.empty(len(values))
    for index, value in enumerate(sorted_values):
        distances = (sorted_values[starts[index]:stops[index]] - value) / width
        kernels = np.exp(-0.5 * distances ** 2) / np.sqrt(2 * np.pi)
        densities[order[index]] = kernels.sum() / width / len(values)
    return densities


def reciprocal_weights(densities, alpha):
    weights = np.asarray(1 / np.power(densities, alpha), dtype="float32")
    total = weights.sum()
    if abs(total - len(weights)) > 1e-3 * len(weights):
        weights *= len(weights) / total
    return weights


def fit_gate_candidate(
    gate,
    features,
    targets,
    labels,
    cache,
    indices,
    candidate,
    epochs,
    settings,
    callbacks=None,
    validation_indices=None,
    densities=None,
):
    """Fit one gate while keeping its classification/regression objective explicit."""

    fit_arguments = {
        "epochs": epochs,
        "batch_size": settings.training_batch_size,
        "callbacks": callbacks or [],
        "shuffle": False,
        "verbose": 0,
    }

    if settings.regression_gate:
        trainer = build_cached_gate_trainer(gate, features.shape[1], settings)
        sample_weights = reciprocal_weights(densities, candidate)

        trainer.fit(
            cache.training_inputs(features, indices, settings),
            targets[indices],
            sample_weight=sample_weights,
            **fit_arguments,
        )
    else:
        # Keep classification validation_loss reporting in addition to the
        # AORE callback. Selection continues to use the ensemble's AORE.
        if validation_indices is not None:
            fit_arguments["validation_data"] = (
                features[validation_indices], labels[validation_indices]
            )

        gate.fit(
            features[indices],
            labels[indices],
            class_weight={0: 1.0, 1: float(candidate)},
            **fit_arguments,
        )


def select_and_train_gate(features, targets, region_labels, cache, settings):
    labels = np.asarray(region_labels).reshape(-1).astype("int32")

    if np.min(np.bincount(labels, minlength=2)) < settings.fold_count:
        raise ValueError("Each region needs at least one validation row per fold.")

    splitter = StratifiedKFold(
        n_splits=settings.fold_count, shuffle=True, random_state=settings.seed
    )
    folds = []

    for training_indices, validation_indices in splitter.split(features, labels):
        densities = (
            compute_sample_densities(targets[training_indices])
            if settings.regression_gate else None
        )
        folds.append((training_indices, validation_indices, densities))

    candidates = (
        settings.alpha_candidates
        if settings.regression_gate else settings.rare_class_weight_candidates
    )

    best_aore = np.inf
    best_candidate = None
    best_epoch_count = None

    for candidate in candidates:
        fold_aores = []
        fold_epochs = []

        for fold_number, (training_indices, validation_indices, densities) in enumerate(
            folds, start=1
        ):
            keras.backend.clear_session()
            set_global_determinism(settings.seed + 100 + fold_number)
            gate = build_gate(features.shape[1], settings)

            validation_cache = ExpertPredictionCache(
                cache.predictions[validation_indices],
                cache.density_quality[validation_indices],
            )

            stopping = GateAOREEarlyStopping(
                gate,
                features[validation_indices],
                targets[validation_indices],
                labels[validation_indices],
                validation_cache,
                settings,
            )

            print(f"Gate: candidate={candidate}, fold {fold_number}/{settings.fold_count}")
            fit_gate_candidate(
                gate, features, targets, labels, cache, training_indices,
                candidate, settings.maximum_epochs, settings,
                callbacks=[stopping], validation_indices=validation_indices,
                densities=densities,
            )

            fold_aores.append(stopping.best_aore)
            fold_epochs.append(stopping.best_epoch)
            del gate, stopping, validation_cache
            gc.collect()

        mean_aore = float(np.mean(fold_aores))
        epoch_count = max(1, int(np.round(np.mean(fold_epochs))))
        print(f"  candidate={candidate}: mean AORE={mean_aore:.6f}, epochs={epoch_count}")

        if mean_aore < best_aore:
            best_aore = mean_aore
            best_candidate = candidate
            best_epoch_count = epoch_count

    if best_epoch_count is None:
        raise ValueError("No finite gate validation result was found.")

    keras.backend.clear_session()
    set_global_determinism(settings.seed + 3000)
    gate = build_gate(features.shape[1], settings)

    final_densities = compute_sample_densities(targets) if settings.regression_gate else None
    fit_gate_candidate(
        gate, features, targets, labels, cache, slice(None),
        best_candidate, best_epoch_count, settings, densities=final_densities,
    )

    print(f"Gate: selected candidate={best_candidate}, epochs={best_epoch_count}")
    return gate


def load_sep_c_dataset():
    with ZipFile(DATA_ARCHIVE) as archive:
        with archive.open("sep_10mev_training_pseudo_labeled.csv") as source:
            train = pd.read_csv(source)
        with archive.open("sep_10mev_testing_pseudo_labeled.csv") as source:
            test = pd.read_csv(source)
    if not train.columns.equals(test.columns) or TARGET not in train or len(train.columns) != 23:
        raise ValueError("SEP-C requires 22 features and ln_peak_intensity in the same column order.")

    def split(data):
        x = data.drop(columns=TARGET).to_numpy(dtype="float32")
        y = data[[TARGET]].to_numpy(dtype="float32")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite SEP-C feature or target.")
        return x, y

    train_x, train_y = split(train)
    test_x, test_y = split(test)
    return train_x, train_y, test_x, test_y


def evaluate_gate(gate, settings, features, targets, cache, rows):
    trust = predict_in_batches(gate, features, settings.prediction_batch_size)
    predictions, weights = mix_cached_predictions(
        trust, cache, use_density=settings.density_at_inference, normalize_trust=settings.sigmoid_gate,
    )
    return record_gate_results(settings.run_name, targets, trust, weights, predictions, rows)


def comparison_rows(targets, expert_predictions, densities, target_column=TARGET):
    rows = pd.DataFrame({"test_index": np.arange(len(targets)), target_column: targets})
    for column, expert in enumerate(("common", "rare")):
        rows[f"{expert}_prediction"] = expert_predictions[:, column]
        rows[f"{expert}_density"] = densities[:, column]
    return rows


def record_gate_results(name, targets, trust, weights, predictions, rows):
    for column, expert in enumerate(("common", "rare")):
        rows[f"{name}_gate_{expert}"] = trust[:, column]
        rows[f"{name}_weight_{expert}"] = weights[:, column]
    rows[f"{name}_prediction"] = predictions
    rows[f"{name}_abs_error"] = np.abs(predictions - targets)
    overall, rare, aore = aore_components(targets, predictions, targets >= RARE_THRESHOLD)
    return {"overall_mae": overall, "rare_mae": rare, "aore": aore}


def print_worst_rare_samples(rows, target_column=TARGET):
    rare = rows.loc[rows[target_column] >= RARE_THRESHOLD]
    top = rare.nlargest(5, "regression_abs_error", keep="first")
    print("\nWorst rare test samples ranked by regression ensemble absolute error")
    print("C = common expert, R = rare expert. Error = Prediction - Target.")
    header = (
        f"{'Method':<14} {'Index':>6} {'Target':>10} {'Common':>10} {'Rare':>10} "
        f"{'Gate C':>9} {'Gate R':>9} {'Quality C':>10} {'Quality R':>10} "
        f"{'Weight C':>10} {'Weight R':>10} {'Prediction':>11} {'Error':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in top.to_dict("records"):
        for method in ("regression", "classification"):
            prediction = row[f"{method}_prediction"]
            print(
                f"{method.capitalize():<14} {row['test_index']:6d} {row[target_column]:10.4f} "
                f"{row['common_prediction']:10.4f} {row['rare_prediction']:10.4f} "
                f"{row[f'{method}_gate_common']:9.4f} {row[f'{method}_gate_rare']:9.4f} "
                f"{row['common_density']:10.4f} {row['rare_density']:10.4f} "
                f"{row[f'{method}_weight_common']:10.4f} {row[f'{method}_weight_rare']:10.4f} "
                f"{prediction:11.4f} {prediction - row[target_column]:10.4f}"
            )


def print_comparison(results, rows, target_column=TARGET):
    print(f"\n{'Gate':<16} {'Overall MAE':>12} {'Rare MAE':>12} {'Test AORE':>12}")
    for name, metrics in results.items():
        print(f"{name:<16} {metrics['overall_mae']:12.6f} {metrics['rare_mae']:12.6f} "
              f"{metrics['aore']:12.6f}")
    print_worst_rare_samples(rows, target_column)


def plot_gate_predictions(rows, target_column, directory):
    from imbal.regression import plot_true_vs_predictions

    actual = rows[target_column].to_numpy()
    for name in ("classification", "regression"):
        path = directory / f"{name}_gate_predicted_vs_actual.png"
        plot_true_vs_predictions(
            actual, rows[f"{name}_prediction"].to_numpy(),
            title=f"{name.capitalize()} gate ({target_column})",
            save_figure=path,
        )
        print(f"Saved prediction plot to {path}", flush=True)


def saved_model_directory(runs, selected_directory):
    if selected_directory is not None:
        return HERE / Path(selected_directory)

    for directory in sorted(runs.glob("????????-??????-??????"), reverse=True):
        if all((directory / f"{gate}_ensemble.keras").is_file()
               for gate in ("classification", "regression")):
            return directory
    raise FileNotFoundError(
        f"No saved pair of ensembles in {runs}. Set LOAD_SAVED_MODEL = False to train "
        "a pair, or set SAVED_MODEL_DIRECTORY to a folder containing both ensembles."
    )


def compare_saved_run(directory, features, targets, target_column=TARGET, *, show_visualizations=False):
    """Read both full ensembles, including their frozen experts and KDEs."""
    print(f"Evaluating saved models in {directory}", flush=True)
    results = {}
    rows = None
    for name in ("classification", "regression"):
        path = directory / f"{name}_ensemble.keras"
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved ensemble: {path}")
        model = keras.models.load_model(path, compile=False)
        probe = keras.Model(model.inputs, [
            model.get_layer("expert_predictions").output,
            model.get_layer("expert_densities").output,
            model.get_layer("density_weighted_gate").input[0],
            model.get_layer("normalized_expert_weights").output,
            model.output,
        ])
        predictions, densities, trust, weights, final = probe.predict(features, batch_size=128, verbose=0)
        if rows is None:
            rows = comparison_rows(targets, predictions, densities, target_column)
        else:
            # The paired table prints one set of expert values for both gates.
            for column, expert in enumerate(("common", "rare")):
                np.testing.assert_array_equal(predictions[:, column], rows[f"{expert}_prediction"])
                np.testing.assert_array_equal(densities[:, column], rows[f"{expert}_density"])
        results[name] = record_gate_results(name, targets, trust, weights, final.ravel(), rows)
    print_comparison(results, rows, target_column)
    if show_visualizations:
        plot_gate_predictions(rows, target_column, directory)


def run_gate_comparison(
    train_x, train_y, test_x, test_y, *, target_column, runs_directory,
    load_saved_model, selected_directory, show_visualizations=False,
):
    if load_saved_model:
        directory = saved_model_directory(runs_directory, selected_directory)
        compare_saved_run(directory, test_x, test_y.ravel(), target_column,
                          show_visualizations=show_visualizations)
        return

    classification = ExperimentSettings(run_name="classification")
    regression = replace(
        classification, run_name="regression", regression_gate=True,
        sigmoid_gate=True, density_during_gate_training=True,
    )
    set_global_determinism(classification.seed)
    regions = (train_y.ravel() >= RARE_THRESHOLD).astype("int32")

    output = runs_directory / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    print(f"Saving this run to {output}", flush=True)

    # The classification settings select unweighted MSE + PCC, five-fold
    # validation-loss epoch selection and a final fit in each expert's region.
    common, rare = train_experts(train_x, train_y, regions, classification)
    common.save(output / "common_expert.keras")
    rare.save(output / "rare_expert.keras")

    profiles = [build_density_profile(train_y[regions == i], classification) for i in (0, 1)]
    # Both gates get the very same cached values. Use the TensorFlow KDE
    # calculation used by the exported ensembles for both splits.
    train_cache = create_prediction_cache(common, rare, train_x, profiles, regression)
    test_cache = create_prediction_cache(common, rare, test_x, profiles, regression)

    # CSV data-row indices (zero-based, excluding the header) identify the
    # same samples across gates even after filtering and ranking.
    targets = test_y.ravel()
    rows = comparison_rows(targets, test_cache.predictions, test_cache.density_quality, target_column)

    results = {}
    for settings in (classification, regression):
        name = settings.run_name
        print(f"Selecting the {name} gate by validation AORE", flush=True)
        gate = select_and_train_gate(train_x, train_y, regions, train_cache, settings)
        gate.save(output / f"{name}_gate.keras")
        ensemble = build_ensemble(common, rare, gate, profiles, settings)
        ensemble.save(output / f"{name}_ensemble.keras")
        results[name] = evaluate_gate(gate, settings, test_x, targets, test_cache, rows)

    (output / "test_metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print_comparison(results, rows, target_column)
    if show_visualizations:
        plot_gate_predictions(rows, target_column, output)


def main():
    run_gate_comparison(
        *load_sep_c_dataset(), target_column=TARGET, runs_directory=HERE / "runs",
        load_saved_model=LOAD_SAVED_MODEL, selected_directory=SAVED_MODEL_DIRECTORY,
        show_visualizations=SHOW_VISUALIZATIONS,
    )


if __name__ == "__main__":
    main()

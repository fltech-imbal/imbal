import os
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold

import imbal


TARGET_COLUMN = "ln_peak_intensity"
RAW_TARGET_COLUMN = "peak_intensity"
NON_FEATURE_COLUMNS = [
    "source_row",
    "SEP_onset_time",
    "CME_DONKI_time",
    "CME_CDAW_time",
]
RARE_THRESHOLD = np.log(10)
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(SCRIPT_DIRECTORY, "sep_10mev_training_raw.csv")
TEST_DATA_PATH = os.path.join(SCRIPT_DIRECTORY, "sep_10mev_testing_raw.csv")
EXPECTED_INPUT_COUNT = 22
MAX_EPOCHS = 500
PATIENCE = 100
BATCH_SIZE = 32
MIN_KDE_BANDWIDTH = 0.05
DENSITY_EPSILON = 1e-7

# Saved-model reuse. Give each experiment/script its own folder so models from
# different trust approaches cannot overwrite one another.
LOAD_SAVED_MODEL = True
MODEL_RUN_NAME = "density_trust_sep_c"
MODEL_ROOT_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "saved_models")
MODEL_DIRECTORY = os.path.join(MODEL_ROOT_DIRECTORY, MODEL_RUN_NAME)
COMMON_EXPERT_PATH = os.path.join(MODEL_DIRECTORY, "common_expert.keras")
RARE_EXPERT_PATH = os.path.join(MODEL_DIRECTORY, "rare_expert.keras")
GATE_PATH = os.path.join(MODEL_DIRECTORY, "gate.keras")
ENSEMBLE_PATH = os.path.join(MODEL_DIRECTORY, "ensemble.keras")


def set_global_determinism(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_sep_c_dataset(
    train_data_path=TRAIN_DATA_PATH,
    test_data_path=TEST_DATA_PATH,
    target_column=TARGET_COLUMN,
    rare_threshold=RARE_THRESHOLD,
    expected_input_count=EXPECTED_INPUT_COUNT,
):
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    for split_name, data in (("training", train_data), ("testing", test_data)):
        required_columns = [target_column, RAW_TARGET_COLUMN, *NON_FEATURE_COLUMNS]
        missing_columns = [
            column for column in required_columns if column not in data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The {split_name} data is missing required columns: "
                f"{missing_columns}."
            )

    # Keep ln_peak_intensity as the regression target. Remove the original
    # peak intensity, row/time metadata, and target from the model inputs.
    columns_to_drop = [
        target_column,
        RAW_TARGET_COLUMN,
        *NON_FEATURE_COLUMNS,
    ]
    training_feature_data = train_data.drop(columns=columns_to_drop)
    testing_feature_data = test_data.drop(columns=columns_to_drop)

    if list(training_feature_data.columns) != list(testing_feature_data.columns):
        raise ValueError(
            "Training and testing feature columns must match in the same order."
        )

    raw_training_features = training_feature_data.to_numpy(dtype="float32")
    raw_testing_features = testing_feature_data.to_numpy(dtype="float32")
    training_features, testing_features = sign_preserving_min_max_scale(
        raw_training_features, raw_testing_features
    )
    training_targets = train_data[target_column].to_numpy(
        dtype="float32"
    ).reshape(-1, 1)
    testing_targets = test_data[target_column].to_numpy(
        dtype="float32"
    ).reshape(-1, 1)

    for split_name, features in (
        ("training", training_features),
        ("testing", testing_features),
    ):
        if features.shape[1] != expected_input_count:
            raise ValueError(
                f"Expected {expected_input_count} input features, but found "
                f"{features.shape[1]} in the {split_name} data."
            )

    training_region_labels = (training_targets >= rare_threshold).astype(
        "int32"
    )
    testing_region_labels = (testing_targets >= rare_threshold).astype("int32")

    all_features = np.concatenate([training_features, testing_features], axis=0)
    all_targets = np.concatenate([training_targets, testing_targets], axis=0)

    return (
        training_features,
        testing_features,
        training_targets,
        testing_targets,
        training_region_labels,
        testing_region_labels,
        all_features,
        all_targets,
    )


def sign_preserving_min_max_scale(training_features, testing_features):
    """Scale each input feature by sign using training-only bounds.

    Negative values are mapped to [-1, 0), positive values to (0, 1], and
    zeros remain exactly zero. Test values outside the training range are
    clipped so the transformed inputs remain within [-1, 1].
    """
    training_features = np.asarray(training_features, dtype="float32")
    testing_features = np.asarray(testing_features, dtype="float32")

    if not np.all(np.isfinite(training_features)):
        raise ValueError("Training features contain NaN or infinite values.")
    if not np.all(np.isfinite(testing_features)):
        raise ValueError("Testing features contain NaN or infinite values.")

    negative_scales = np.max(
        np.where(training_features < 0, -training_features, 0.0), axis=0
    )
    positive_scales = np.max(
        np.where(training_features > 0, training_features, 0.0), axis=0
    )

    def transform(features):
        scaled = np.zeros_like(features, dtype="float32")
        negative_mask = features < 0
        positive_mask = features > 0

        safe_negative_scales = np.where(negative_scales > 0, negative_scales, 1.0)
        safe_positive_scales = np.where(positive_scales > 0, positive_scales, 1.0)
        scaled = np.where(
            negative_mask,
            features / safe_negative_scales,
            scaled,
        )
        scaled = np.where(
            positive_mask,
            features / safe_positive_scales,
            scaled,
        )
        return np.clip(scaled, -1.0, 1.0).astype("float32")

    return transform(training_features), transform(testing_features)


def build_expert(expert_name, input_count=EXPECTED_INPUT_COUNT):
    """Build one regressor to specialize in a single target range."""
    inputs = keras.Input(shape=(input_count,), name=f"{expert_name}_input")
    hidden = layers.Dense(
        18, activation="relu", name=f"{expert_name}_dense_1"
    )(inputs)
    hidden = layers.Dense(
        12, activation="relu", name=f"{expert_name}_dense_2"
    )(hidden)
    hidden = layers.Dense(
        8, activation="relu", name=f"{expert_name}_dense_3"
    )(hidden)
    representation = layers.Dense(
        6, activation="relu", name=f"{expert_name}_representation"
    )(hidden)
    output = layers.Dense(1, name=f"{expert_name}_prediction")(representation)
    return keras.Model(inputs, output, name=expert_name)


def compile_expert(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )


def build_gate(input_count=EXPECTED_INPUT_COUNT):
    """Predict [P(common | x), P(rare | x)] from the input features."""
    inputs = keras.Input(shape=(input_count,), name="gate_input")
    hidden = layers.Dense(18, activation="relu", name="gate_dense_1")(inputs)
    hidden = layers.Dense(12, activation="relu", name="gate_dense_2")(hidden)
    hidden = layers.Dense(8, activation="relu", name="gate_representation")(
        hidden
    )
    probabilities = layers.Dense(
        2, activation="softmax", name="gate_probabilities"
    )(hidden)
    return keras.Model(inputs, probabilities, name="gate")


def compile_gate(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def make_gate_class_weights(rare_class_weight):
    """Weight common samples as 1.0 and rare samples by the candidate value."""
    return {
        0: 1.0,
        1: float(rare_class_weight),
    }


def select_kde_bandwidth(targets, minimum_bandwidth=MIN_KDE_BANDWIDTH):
    """Silverman's rule with a floor for constant or nearly constant targets."""
    values = np.asarray(targets, dtype="float64").reshape(-1)
    if len(values) < 2:
        return float(minimum_bandwidth)
    estimated = 1.06 * np.std(values, ddof=1) * (len(values) ** (-1.0 / 5.0))
    if not np.isfinite(estimated):
        estimated = 0.0
    return float(max(estimated, minimum_bandwidth))


def gaussian_kde_density(query_values, training_targets, bandwidth):
    """Evaluate a one-dimensional Gaussian KDE in manageable chunks."""
    queries = np.asarray(query_values, dtype="float64").reshape(-1)
    samples = np.asarray(training_targets, dtype="float64").reshape(-1)
    if len(samples) == 0:
        raise ValueError("A density profile cannot be built without training targets.")
    normalizer = len(samples) * bandwidth * np.sqrt(2.0 * np.pi)
    densities = np.empty(len(queries), dtype="float64")
    for start in range(0, len(queries), 1024):
        stop = min(start + 1024, len(queries))
        differences = (queries[start:stop, None] - samples[None, :]) / bandwidth
        densities[start:stop] = np.sum(np.exp(-0.5 * differences ** 2), axis=1) / normalizer
    return densities


def build_density_profile(expert_targets):
    """Build an independently normalized KDE from one expert's assigned targets."""
    targets = np.asarray(expert_targets, dtype="float32").reshape(-1)
    bandwidth = select_kde_bandwidth(targets)
    training_densities = gaussian_kde_density(targets, targets, bandwidth)
    maximum_density = float(np.max(training_densities))
    if not np.isfinite(maximum_density) or maximum_density <= 0.0:
        raise ValueError("The KDE maximum density must be positive and finite.")
    return {
        "training_targets": targets,
        "bandwidth": bandwidth,
        "maximum_density": maximum_density,
    }


def evaluate_normalized_density(predictions, profile):
    raw_density = gaussian_kde_density(
        predictions, profile["training_targets"], profile["bandwidth"]
    )
    normalized = raw_density / profile["maximum_density"]
    return np.clip(normalized, DENSITY_EPSILON, 1.0).astype("float32")


def apply_density_to_gate_probabilities(common_predictions, rare_predictions,
                                        gate_probabilities, common_profile,
                                        rare_profile):
    densities = np.column_stack([
        evaluate_normalized_density(common_predictions, common_profile),
        evaluate_normalized_density(rare_predictions, rare_profile),
    ])
    adjusted = np.asarray(gate_probabilities) * densities
    return adjusted / np.maximum(
        np.sum(adjusted, axis=1, keepdims=True), DENSITY_EPSILON
    )


@keras.utils.register_keras_serializable(package="SEPDensityTrust")
class NormalizedGaussianKDE(layers.Layer):
    """Evaluate a saved one-dimensional KDE and return relative density."""

    def __init__(self, training_targets, bandwidth, maximum_density, **kwargs):
        super().__init__(**kwargs)
        self.training_targets_list = np.asarray(
            training_targets, dtype="float32"
        ).reshape(-1).tolist()
        self.bandwidth = float(bandwidth)
        self.maximum_density = float(maximum_density)

    def call(self, predictions):
        samples = tf.constant(self.training_targets_list, dtype=predictions.dtype)
        differences = (
            tf.reshape(predictions, [-1, 1]) - tf.reshape(samples, [1, -1])
        ) / tf.cast(self.bandwidth, predictions.dtype)
        kernel_sum = tf.reduce_sum(tf.exp(-0.5 * tf.square(differences)), axis=1)
        normalizer = tf.cast(
            len(self.training_targets_list) * self.bandwidth * np.sqrt(2.0 * np.pi),
            predictions.dtype,
        )
        raw_density = kernel_sum / normalizer
        normalized = raw_density / tf.cast(self.maximum_density, predictions.dtype)
        return tf.reshape(tf.clip_by_value(normalized, DENSITY_EPSILON, 1.0), [-1, 1])

    def get_config(self):
        config = super().get_config()
        config.update({
            "training_targets": self.training_targets_list,
            "bandwidth": self.bandwidth,
            "maximum_density": self.maximum_density,
        })
        return config


@keras.utils.register_keras_serializable(package="SEPDensityTrust")
class NormalizeExpertWeights(layers.Layer):
    def call(self, values):
        denominator = tf.maximum(
            tf.reduce_sum(values, axis=1, keepdims=True), DENSITY_EPSILON
        )
        return values / denominator


def get_expert_epoch_count_with_kfold(
    features,
    targets,
    expert_name,
    fold_count=5,
    seed=42,
):
    if len(features) < fold_count:
        raise ValueError(
            f"The {expert_name} range has only {len(features)} samples, so "
            f"{fold_count}-fold validation cannot be performed."
        )

    splitter = KFold(n_splits=fold_count, shuffle=True, random_state=seed)
    best_epochs = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(features), start=1
    ):
        tf.keras.backend.clear_session()
        set_global_determinism(seed + fold_number)
        model = build_expert(expert_name, input_count=features.shape[1])
        compile_expert(model)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            mode="min",
            restore_best_weights=True,
        )
        history = model.fit(
            features[train_indices],
            targets[train_indices],
            validation_data=(
                features[validation_indices],
                targets[validation_indices],
            ),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            shuffle=False,
            verbose=0,
        )
        best_epochs.append(int(np.argmin(history.history["val_loss"]) + 1))

    return max(1, int(np.round(np.mean(best_epochs))))



class GateEnsembleAOREEarlyStopping(keras.callbacks.Callback):
    """Early-stop a gate using validation AORE of the full fixed-expert ensemble."""

    def __init__(
        self,
        common_expert,
        rare_expert,
        validation_features,
        validation_targets,
        validation_region_labels,
        common_density_profile,
        rare_density_profile,
        patience=PATIENCE,
    ):
        super().__init__()
        self.common_expert = common_expert
        self.rare_expert = rare_expert
        self.validation_features = validation_features
        self.validation_targets = np.asarray(validation_targets).reshape(-1)
        self.validation_region_labels = (
            np.asarray(validation_region_labels).reshape(-1).astype(bool)
        )
        self.patience = patience
        self.common_density_profile = common_density_profile
        self.rare_density_profile = rare_density_profile

        self.best_aore = np.inf
        self.best_epoch = 1
        self.best_weights = None
        self.wait = 0

        # Experts are fixed across every candidate gate and every fold.
        self.common_predictions = np.asarray(
            common_expert(validation_features, training=False)
        ).reshape(-1)
        self.rare_predictions = np.asarray(
            rare_expert(validation_features, training=False)
        ).reshape(-1)

        # Expert predictions and KDE profiles are fixed throughout gate training.
        # Cache their density qualities once for this validation fold.
        self.common_quality = evaluate_normalized_density(
            self.common_predictions, self.common_density_profile
        )
        self.rare_quality = evaluate_normalized_density(
            self.rare_predictions, self.rare_density_profile
        )
        self.quality_values = np.column_stack(
            [self.common_quality, self.rare_quality]
        )

    def on_epoch_end(self, epoch, logs=None):
        gate_probabilities = np.asarray(
            self.model(self.validation_features, training=False)
        )
        # Only the gate probabilities change from epoch to epoch. Reuse the
        # cached KDE qualities rather than recomputing both KDEs every epoch.
        unnormalized_weights = gate_probabilities * self.quality_values
        adjusted_probabilities = unnormalized_weights / np.maximum(
            np.sum(unnormalized_weights, axis=1, keepdims=True),
            DENSITY_EPSILON,
        )
        ensemble_predictions = (
            adjusted_probabilities[:, 0] * self.common_predictions
            + adjusted_probabilities[:, 1] * self.rare_predictions
        )

        rare_mask = self.validation_region_labels
        if not np.any(rare_mask):
            raise ValueError(
                "No rare validation samples were found when computing gate AORE."
            )

        overall_mae = mean_absolute_error(
            self.validation_targets,
            ensemble_predictions,
        )
        rare_mae = mean_absolute_error(
            self.validation_targets[rare_mask],
            ensemble_predictions[rare_mask],
        )
        current_aore = (overall_mae + rare_mae) / 2.0

        if logs is not None:
            logs["val_ensemble_aore"] = current_aore

        if current_aore < self.best_aore:
            self.best_aore = current_aore
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def evaluate_gate_weight_with_kfold(
    features,
    targets,
    region_labels,
    common_expert,
    rare_expert,
    common_density_profile,
    rare_density_profile,
    rare_class_weight,
    fold_count=5,
    seed=42,
):
    """Return mean validation ensemble AORE and mean best gate epoch."""
    labels = np.asarray(region_labels).reshape(-1).astype("int32")
    class_counts = np.bincount(labels, minlength=2)
    if np.min(class_counts) < fold_count:
        raise ValueError(
            "Each target range must contain at least fold_count samples for "
            "stratified gate validation."
        )

    splitter = StratifiedKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=seed,
    )

    fold_aores = []
    best_epochs = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(features, labels),
        start=1,
    ):
        print(
            f"  Rare class weight {rare_class_weight:g}: "
            f"gate fold {fold_number}/{fold_count}"
        )
        tf.keras.backend.clear_session()
        set_global_determinism(seed + 100 + fold_number)

        gate = build_gate(input_count=features.shape[1])
        compile_gate(gate)

        aore_early_stopping = GateEnsembleAOREEarlyStopping(
            common_expert=common_expert,
            rare_expert=rare_expert,
            validation_features=features[validation_indices],
            validation_targets=targets[validation_indices],
            validation_region_labels=labels[validation_indices],
            common_density_profile=common_density_profile,
            rare_density_profile=rare_density_profile,
            patience=PATIENCE,
        )

        gate.fit(
            features[train_indices],
            labels[train_indices],
            validation_data=(
                features[validation_indices],
                labels[validation_indices],
            ),
            class_weight=make_gate_class_weights(rare_class_weight),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[aore_early_stopping],
            shuffle=False,
            verbose=0,
        )

        fold_aores.append(aore_early_stopping.best_aore)
        best_epochs.append(aore_early_stopping.best_epoch)

    mean_validation_aore = float(np.mean(fold_aores))
    mean_best_epoch = max(1, int(np.round(np.mean(best_epochs))))

    return mean_validation_aore, mean_best_epoch

def train_expert(features, targets, expert_name, epoch_count, seed):
    tf.keras.backend.clear_session()
    set_global_determinism(seed)
    model = build_expert(expert_name, input_count=features.shape[1])
    compile_expert(model)
    model.fit(
        features,
        targets,
        epochs=epoch_count,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )
    return model


def train_gate(
    features,
    region_labels,
    rare_class_weight,
    epoch_count,
    seed,
):
    labels = np.asarray(region_labels).reshape(-1).astype("int32")
    tf.keras.backend.clear_session()
    set_global_determinism(seed)
    model = build_gate(input_count=features.shape[1])
    compile_gate(model)
    model.fit(
        features,
        labels,
        class_weight=make_gate_class_weights(rare_class_weight),
        epochs=epoch_count,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )
    return model


def build_gated_ensemble(common_expert, rare_expert, gate,
                         common_density_profile, rare_density_profile):
    """Combine experts using gate probability times within-expert density."""
    common_expert.trainable = False
    rare_expert.trainable = False
    gate.trainable = False

    inputs = keras.Input(shape=common_expert.input_shape[1:], name="features")
    common_prediction = common_expert(inputs)
    rare_prediction = rare_expert(inputs)
    gate_probabilities = gate(inputs)

    common_density = NormalizedGaussianKDE(
        **common_density_profile, name="common_expert_density"
    )(common_prediction)
    rare_density = NormalizedGaussianKDE(
        **rare_density_profile, name="rare_expert_density"
    )(rare_prediction)
    expert_densities = layers.Concatenate(name="expert_densities")(
        [common_density, rare_density]
    )
    density_weighted_gate = layers.Multiply(name="density_weighted_gate")(
        [gate_probabilities, expert_densities]
    )
    normalized_weights = NormalizeExpertWeights(name="normalized_expert_weights")(
        density_weighted_gate
    )

    expert_predictions = layers.Concatenate(name="expert_predictions")(
        [common_prediction, rare_prediction]
    )
    final_prediction = layers.Dot(axes=1, name="gated_prediction")(
        [expert_predictions, normalized_weights]
    )

    return keras.Model(inputs, final_prediction, name="two_expert_gated_ensemble")


def evaluate_regressor_with_aore_components(
    model,
    evaluation_features,
    evaluation_targets,
    evaluation_region_labels,
):
    predictions = np.asarray(model(evaluation_features, training=False)).reshape(-1)
    targets = np.asarray(evaluation_targets).reshape(-1)
    rare_mask = np.asarray(evaluation_region_labels).reshape(-1).astype(bool)

    if not np.any(rare_mask):
        raise ValueError("No rare samples were found when computing rare MAE.")

    overall_mae = mean_absolute_error(targets, predictions)
    rare_mae = mean_absolute_error(targets[rare_mask], predictions[rare_mask])
    return overall_mae, rare_mae, (overall_mae + rare_mae) / 2.0


def print_gate_summary(gate, features, region_labels, split_name):
    probabilities = np.asarray(gate(features, training=False))
    predicted_labels = np.argmax(probabilities, axis=1)
    true_labels = np.asarray(region_labels).reshape(-1).astype("int32")
    accuracy = np.mean(predicted_labels == true_labels)
    rare_mask = true_labels == 1
    common_mask = true_labels == 0

    print(f"{split_name} gate accuracy: {accuracy:.4f}")
    print(
        f"{split_name} mean P(rare) for true common samples: "
        f"{np.mean(probabilities[common_mask, 1]):.4f}"
    )
    print(
        f"{split_name} mean P(rare) for true rare samples: "
        f"{np.mean(probabilities[rare_mask, 1]):.4f}"
    )


def print_density_weight_summary(common_expert, rare_expert, gate, features,
                                 region_labels, common_profile, rare_profile,
                                 split_name):
    common_predictions = np.asarray(
        common_expert(features, training=False)
    ).reshape(-1)
    rare_predictions = np.asarray(
        rare_expert(features, training=False)
    ).reshape(-1)
    gate_probabilities = np.asarray(gate(features, training=False))
    adjusted = apply_density_to_gate_probabilities(
        common_predictions, rare_predictions, gate_probabilities,
        common_profile, rare_profile
    )
    labels = np.asarray(region_labels).reshape(-1).astype("int32")
    common_mask = labels == 0
    rare_mask = labels == 1
    print(
        f"{split_name} mean density-adjusted P(rare) for true common samples: "
        f"{np.mean(adjusted[common_mask, 1]):.4f}"
    )
    print(
        f"{split_name} mean density-adjusted P(rare) for true rare samples: "
        f"{np.mean(adjusted[rare_mask, 1]):.4f}"
    )



def print_expert_trust_analysis(common_expert, rare_expert, gate, features,
                                targets, common_profile, rare_profile,
                                split_name):
    """Diagnose gate and density trust by true target region.

    Common:   y < 0
    Elevated: 0 <= y < ln(10)  (still expected to favor the common expert)
    Rare:     y >= ln(10)

    For each region, report how often the gate and density quality terms favor
    the expected expert, plus their four-way agreement breakdown. Also print
    per-instance diagnostics for elevated and rare samples.
    """
    common_predictions = np.asarray(
        common_expert(features, training=False)
    ).reshape(-1)
    rare_predictions = np.asarray(
        rare_expert(features, training=False)
    ).reshape(-1)
    gate_probabilities = np.asarray(gate(features, training=False))
    true_targets = np.asarray(targets).reshape(-1)

    common_quality = evaluate_normalized_density(
        common_predictions, common_profile
    )
    rare_quality = evaluate_normalized_density(
        rare_predictions, rare_profile
    )

    quality_values = np.column_stack([common_quality, rare_quality])
    unnormalized_weights = gate_probabilities * quality_values
    normalized_weights = unnormalized_weights / np.maximum(
        np.sum(unnormalized_weights, axis=1, keepdims=True),
        DENSITY_EPSILON,
    )
    final_predictions = (
        normalized_weights[:, 0] * common_predictions
        + normalized_weights[:, 1] * rare_predictions
    )

    common_mask = true_targets < 0.0
    elevated_mask = (true_targets >= 0.0) & (true_targets < RARE_THRESHOLD)
    rare_mask = true_targets >= RARE_THRESHOLD

    def summarize_region(region_name, mask, expected_expert):
        indices = np.flatnonzero(mask)
        count = len(indices)

        print(f"\n{split_name} {region_name.upper()} TRUST ANALYSIS (n={count})")
        if count == 0:
            print("No samples in this region.")
            return

        if expected_expert == "common":
            gate_correct = (
                gate_probabilities[indices, 0]
                > gate_probabilities[indices, 1]
            )
            quality_correct = (
                common_quality[indices] > rare_quality[indices]
            )
            expected_label = "common"
        else:
            gate_correct = (
                gate_probabilities[indices, 1]
                > gate_probabilities[indices, 0]
            )
            quality_correct = (
                rare_quality[indices] > common_quality[indices]
            )
            expected_label = "rare"

        both_correct = gate_correct & quality_correct
        gate_only = gate_correct & ~quality_correct
        quality_only = ~gate_correct & quality_correct
        both_wrong = ~gate_correct & ~quality_correct

        def report(label, values):
            number = int(np.sum(values))
            percent = 100.0 * number / count
            print(f"{label}: {number}/{count} = {percent:.2f}%")

        print(f"Expected expert: {expected_label}")
        report(f"Gate favors {expected_label}", gate_correct)
        report(f"Density quality favors {expected_label}", quality_correct)

        print("\nGate / Density agreement:")
        report("  Gate correct, density correct", both_correct)
        report("  Gate correct, density wrong", gate_only)
        report("  Gate wrong, density correct", quality_only)
        report("  Gate wrong, density wrong", both_wrong)

    summarize_region("common (y < 0)", common_mask, "common")
    summarize_region(
        "elevated (0 <= y < ln(10))", elevated_mask, "common"
    )
    summarize_region("rare (y >= ln(10))", rare_mask, "rare")

    def print_instance_table(region_name, mask):
        indices = np.flatnonzero(mask)
        print(f"\n{split_name} {region_name.upper()} PER-INSTANCE TRUST")
        if len(indices) == 0:
            print("No samples in this region.")
            return

        header = (
            f"{'Idx':>5} {'True y':>10} {'CommonPred':>12} {'RarePred':>12} "
            f"{'g(C)':>9} {'g(R)':>9} {'q(C)':>9} {'q(R)':>9} "
            f"{'w(C)':>9} {'w(R)':>9} {'FinalPred':>12} {'Delta':>10}"
        )
        print(header)
        print("-" * len(header))

        for index in indices:
            print(
                f"{index:5d} "
                f"{true_targets[index]:10.4f} "
                f"{common_predictions[index]:12.4f} "
                f"{rare_predictions[index]:12.4f} "
                f"{gate_probabilities[index, 0]:9.4f} "
                f"{gate_probabilities[index, 1]:9.4f} "
                f"{common_quality[index]:9.4f} "
                f"{rare_quality[index]:9.4f} "
                f"{normalized_weights[index, 0]:9.4f} "
                f"{normalized_weights[index, 1]:9.4f} "
                f"{final_predictions[index]:12.4f} "
                f"{(final_predictions[index] - true_targets[index]):10.4f}"
            )

    # Ordinary common samples can be numerous, so the detailed rows focus on
    # the two regions of special interest requested for this diagnostic.
    print_instance_table("elevated samples", elevated_mask)
    print_instance_table("rare samples", rare_mask)


def run_two_expert_gated_ensemble(seed=42):
    set_global_determinism(seed)
    (
        training_features,
        testing_features,
        training_targets,
        testing_targets,
        training_region_labels,
        testing_region_labels,
        all_features,
        all_targets,
    ) = load_sep_c_dataset()

    common_mask = training_region_labels.reshape(-1) == 0
    rare_mask = training_region_labels.reshape(-1) == 1
    common_features = training_features[common_mask]
    common_targets = training_targets[common_mask]
    rare_features = training_features[rare_mask]
    rare_targets = training_targets[rare_mask]

    print(f"Training samples: {len(training_features)}")
    print(f"Common training samples: {len(common_features)}")
    print(f"Rare training samples: {len(rare_features)}")
    print(f"Testing samples: {len(testing_features)}")

    # Density profiles are deterministic functions of the current training targets,
    # so they can be reconstructed when reusing the saved neural networks.
    common_density_profile = build_density_profile(common_targets)
    rare_density_profile = build_density_profile(rare_targets)
    print("\nDensity Trust Profiles")
    print(f"Common KDE bandwidth: {common_density_profile['bandwidth']:.6f}")
    print(f"Rare KDE bandwidth: {rare_density_profile['bandwidth']:.6f}")

    if LOAD_SAVED_MODEL:
        required_model_paths = [COMMON_EXPERT_PATH, RARE_EXPERT_PATH, GATE_PATH]
        missing_model_paths = [
            path for path in required_model_paths if not os.path.exists(path)
        ]
        if missing_model_paths:
            raise FileNotFoundError(
                "LOAD_SAVED_MODEL=True, but these saved model files are missing: "
                f"{missing_model_paths}. Run once with LOAD_SAVED_MODEL=False first."
            )

        print(f"\nLoading saved models from: {MODEL_DIRECTORY}")
        common_expert = keras.models.load_model(COMMON_EXPERT_PATH)
        rare_expert = keras.models.load_model(RARE_EXPERT_PATH)
        gate = keras.models.load_model(GATE_PATH)
        ensemble = build_gated_ensemble(
            common_expert,
            rare_expert,
            gate,
            common_density_profile,
            rare_density_profile,
        )
        print("Saved models loaded; skipping k-fold searches and retraining.")
    else:
        common_epochs = get_expert_epoch_count_with_kfold(
            common_features, common_targets, "common_expert", seed=seed
        )
        rare_epochs = get_expert_epoch_count_with_kfold(
            rare_features, rare_targets, "rare_expert", seed=seed
        )
        print("\nK-fold Epoch Estimates")
        print(f"Common expert epochs: {common_epochs}")
        print(f"Rare expert epochs: {rare_epochs}")

        common_expert = train_expert(
            common_features, common_targets, "common_expert", common_epochs, seed + 1000
        )
        rare_expert = train_expert(
            rare_features, rare_targets, "rare_expert", rare_epochs, seed + 2000
        )

        candidate_rare_class_weights = [
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
        ]
        best_gate_rare_class_weight = None
        best_gate_epoch_count = None
        best_validation_aore = np.inf

        print("\nGate Class-Weight Search")
        for rare_class_weight in candidate_rare_class_weights:
            candidate_validation_aore, candidate_gate_epochs = (
                evaluate_gate_weight_with_kfold(
                    training_features,
                    training_targets,
                    training_region_labels,
                    common_expert,
                    rare_expert,
                    common_density_profile,
                    rare_density_profile,
                    rare_class_weight=rare_class_weight,
                    fold_count=5,
                    seed=seed,
                )
            )
            print(
                f"Rare class weight={rare_class_weight}, "
                f"mean best gate epochs={candidate_gate_epochs}, "
                f"mean validation ensemble AORE={candidate_validation_aore:.4f}"
            )
            if candidate_validation_aore < best_validation_aore:
                best_validation_aore = candidate_validation_aore
                best_gate_rare_class_weight = rare_class_weight
                best_gate_epoch_count = candidate_gate_epochs

        print("\nBest Gate Configuration")
        print(f"Rare class weight: {best_gate_rare_class_weight}")
        print(f"Gate epochs: {best_gate_epoch_count}")
        print(
            "Mean k-fold validation ensemble AORE used for selection: "
            f"{best_validation_aore:.4f}"
        )

        gate = train_gate(
            training_features,
            training_region_labels,
            rare_class_weight=best_gate_rare_class_weight,
            epoch_count=best_gate_epoch_count,
            seed=seed + 3000,
        )
        ensemble = build_gated_ensemble(
            common_expert, rare_expert, gate,
            common_density_profile, rare_density_profile,
        )

        os.makedirs(MODEL_DIRECTORY, exist_ok=True)
        common_expert.save(COMMON_EXPERT_PATH)
        rare_expert.save(RARE_EXPERT_PATH)
        gate.save(GATE_PATH)
        ensemble.save(ENSEMBLE_PATH)
        print(f"\nSaved trained models to: {MODEL_DIRECTORY}")

    print("\nGate Results")
    print_gate_summary(gate, training_features, training_region_labels, "Training")
    print_gate_summary(gate, testing_features, testing_region_labels, "Test")
    print_density_weight_summary(
        common_expert, rare_expert, gate, training_features,
        training_region_labels, common_density_profile, rare_density_profile,
        "Training"
    )
    print_density_weight_summary(
        common_expert, rare_expert, gate, testing_features,
        testing_region_labels, common_density_profile, rare_density_profile,
        "Test"
    )


    print("\nTest Expert Trust Diagnostics")
    print_expert_trust_analysis(
        common_expert,
        rare_expert,
        gate,
        testing_features,
        testing_targets,
        common_density_profile,
        rare_density_profile,
        "Test",
    )

    training_metrics = evaluate_regressor_with_aore_components(
        ensemble,
        training_features,
        training_targets,
        training_region_labels,
    )
    testing_metrics = evaluate_regressor_with_aore_components(
        ensemble,
        testing_features,
        testing_targets,
        testing_region_labels,
    )

    print("\nTraining Ensemble Results")
    print(f"Training overall MAE: {training_metrics[0]:.4f}")
    print(f"Training rare MAE: {training_metrics[1]:.4f}")
    print(f"Training AORE: {training_metrics[2]:.4f}")

    print("\nFinal Test Set Results")
    print(f"Test overall MAE: {testing_metrics[0]:.4f}")
    print(f"Test rare MAE: {testing_metrics[1]:.4f}")
    print(f"Test AORE: {testing_metrics[2]:.4f}")

    common_representation = keras.Model(
        common_expert.input,
        common_expert.get_layer("common_expert_representation").output,
        name="common_expert_representation_model",
    )
    rare_representation = keras.Model(
        rare_expert.input,
        rare_expert.get_layer("rare_expert_representation").output,
        name="rare_expert_representation_model",
    )
    gate_representation = keras.Model(
        gate.input,
        gate.get_layer("gate_representation").output,
        name="gate_representation_model",
    )

    imbal.regression.tsne_visualization(
        common_representation, all_features, all_targets
    )
    imbal.regression.tsne_visualization(
        rare_representation, all_features, all_targets
    )
    imbal.regression.tsne_visualization(
        gate_representation, all_features, all_targets
    )

    test_predictions = ensemble.predict(testing_features)
    imbal.regression.plot_true_vs_predictions(testing_targets, test_predictions)

    return ensemble, common_expert, rare_expert, gate


if __name__ == "__main__":
    final_ensemble, common_model, rare_model, gating_model = (
        run_two_expert_gated_ensemble(seed=42)
    )

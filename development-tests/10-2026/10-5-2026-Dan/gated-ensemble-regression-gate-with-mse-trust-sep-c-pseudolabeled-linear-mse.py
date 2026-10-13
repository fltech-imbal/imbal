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
RARE_THRESHOLD = np.log(10)
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(SCRIPT_DIRECTORY, "sep_10mev_training_pseudo_labeled.csv")
TEST_DATA_PATH = os.path.join(SCRIPT_DIRECTORY, "sep_10mev_testing_pseudo_labeled.csv")
EXPECTED_INPUT_COUNT = 22
MAX_EPOCHS = 500
PATIENCE = 100
BATCH_SIZE = 32

LOAD_SAVED_MODEL = False
MODEL_RUN_NAME = "regression_gate_mse_trust_sep_c_pseudolabeled_linear_mse"
MODEL_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "saved_models", MODEL_RUN_NAME)
COMMON_MODEL_PATH = os.path.join(MODEL_DIRECTORY, "common_expert.keras")
RARE_MODEL_PATH = os.path.join(MODEL_DIRECTORY, "rare_expert.keras")
GATE_MODEL_PATH = os.path.join(MODEL_DIRECTORY, "gate.keras")
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIRECTORY, "ensemble.keras")
TRUST_MODE = "mse"
TRUST_BIN_COUNT = 2
TRUST_EPSILON = 1e-7
MAX_MSE = 9.0


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
        if target_column not in data.columns:
            raise ValueError(
                f"The {split_name} pseudolabeled data is missing required "
                f"target column: {target_column}."
            )

    # Pseudolabeled files already contain the 22 normalized model inputs plus
    # ln_peak_intensity. Use the input columns exactly as stored in the files.
    training_feature_data = train_data.drop(columns=[target_column])
    testing_feature_data = test_data.drop(columns=[target_column])

    if list(training_feature_data.columns) != list(testing_feature_data.columns):
        raise ValueError(
            "Training and testing feature columns must match in the same order."
        )

    training_features = training_feature_data.to_numpy(dtype="float32")
    testing_features = testing_feature_data.to_numpy(dtype="float32")

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
        if not np.all(np.isfinite(features)):
            raise ValueError(
                f"The {split_name} pseudolabeled features contain NaN or infinite values."
            )

    training_region_labels = (training_targets >= rare_threshold).astype("int32")
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


ALPHA_CANDIDATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
SUBBIN_BALANCE_TOLERANCE = 0.10


def build_gate(input_count=EXPECTED_INPUT_COUNT):
    """Output learned normalized expert trusts [g(C), g(R)]."""
    inputs = keras.Input(shape=(input_count,), name="gate_input")
    x = layers.Dense(18, activation="relu", name="gate_dense_1")(inputs)
    x = layers.Dense(12, activation="relu", name="gate_dense_2")(x)
    x = layers.Dense(8, activation="relu", name="gate_representation")(x)
    trust = layers.Dense(2, activation="softmax", name="gate_trust")(x)
    return keras.Model(inputs, trust, name="gate")


def reciprocal_weights(targets, alpha):
    y = np.asarray(targets).reshape(-1)
    kde = imbal.regression.fit_kde(y)
    density = imbal.regression.get_sample_densities(y, kde)
    return np.asarray(
        imbal.regression.reciprocal_importance(density, alpha=float(alpha)),
        dtype="float32",
    ).reshape(-1)


def expert_subbin_info(targets):
    y = np.asarray(targets).reshape(-1)
    midpoint = float(np.min(y) + (np.max(y) - np.min(y)) / 2.0)
    low, high = int(np.sum(y <= midpoint)), int(np.sum(y > midpoint))
    difference = abs(low - high) / max(low, high, 1)
    return midpoint, low, high, difference, difference >= SUBBIN_BALANCE_TOLERANCE


def subbin_average_error(targets, predictions, midpoint):
    y = np.asarray(targets).reshape(-1)
    pred = np.asarray(predictions).reshape(-1)
    low, high = y <= midpoint, y > midpoint
    if not np.any(low) or not np.any(high):
        return np.inf
    return (
        mean_absolute_error(y[low], pred[low])
        + mean_absolute_error(y[high], pred[high])
    ) / 2.0


class ExpertSAEEarlyStopping(keras.callbacks.Callback):
    def __init__(self, x_val, y_val, midpoint, patience=PATIENCE):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.midpoint = midpoint
        self.patience = patience
        self.best_sae = np.inf
        self.best_epoch = 1
        self.best_weights = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        sae = subbin_average_error(
            self.y_val, self.model(self.x_val, training=False), self.midpoint
        )
        if sae < self.best_sae:
            self.best_sae = sae
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


def evaluate_expert_alpha_with_kfold(
    features, targets, expert_name, alpha, midpoint, weighted,
    fold_count=5, seed=42
):
    sublabels = (np.asarray(targets).reshape(-1) > midpoint).astype("int32")
    counts = np.bincount(sublabels, minlength=2)
    if np.min(counts) < fold_count:
        raise ValueError(
            f"{expert_name}: each sub-bin needs at least {fold_count} samples "
            "for SAE k-fold validation."
        )

    splitter = StratifiedKFold(
        n_splits=fold_count, shuffle=True, random_state=seed
    )
    saes, epochs = [], []

    for fold, (tr, va) in enumerate(splitter.split(features, sublabels), 1):
        tf.keras.backend.clear_session()
        set_global_determinism(seed + fold)
        model = build_expert(expert_name, features.shape[1])
        compile_expert(model)
        cb = ExpertSAEEarlyStopping(features[va], targets[va], midpoint)
        sw = reciprocal_weights(targets[tr], alpha) if weighted else None
        model.fit(
            features[tr],
            targets[tr],
            sample_weight=sw,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[cb],
            shuffle=False,
            verbose=0,
        )
        saes.append(cb.best_sae)
        epochs.append(cb.best_epoch)

    return float(np.mean(saes)), max(1, int(np.round(np.mean(epochs))))


def select_and_train_expert(features, targets, expert_name, seed):
    midpoint, n1, n2, diff, imbalanced = expert_subbin_info(targets)
    print(
        f"\n{expert_name} sub-bins: midpoint={midpoint:.6f}, "
        f"counts={n1}/{n2}, difference={100 * diff:.2f}%"
    )
    candidates = ALPHA_CANDIDATES if imbalanced else [None]
    best = (np.inf, None, None)

    for alpha in candidates:
        sae, epochs = evaluate_expert_alpha_with_kfold(
            features,
            targets,
            expert_name,
            0.0 if alpha is None else alpha,
            midpoint,
            imbalanced,
            seed=seed,
        )
        print(
            f"  {'regular MSE' if alpha is None else f'alpha={alpha}'}: "
            f"SAE={sae:.4f}, epochs={epochs}"
        )
        if sae < best[0]:
            best = (sae, alpha, epochs)

    _, alpha, epochs = best
    tf.keras.backend.clear_session()
    set_global_determinism(seed + 1000)
    model = build_expert(expert_name, features.shape[1])
    compile_expert(model)
    sw = reciprocal_weights(targets, alpha) if imbalanced else None
    model.fit(
        features,
        targets,
        sample_weight=sw,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )
    print(
        f"  selected alpha: {alpha if imbalanced else 'N/A (balanced)'}; "
        f"selected epochs: {epochs}; SAE={best[0]:.4f}"
    )
    return model, alpha, epochs


def build_expert_trust_profile(expert, features, targets, bin_count=TRUST_BIN_COUNT):
    """Estimate prediction-bin MSE trust from training data only."""
    predictions = np.asarray(expert(features, training=False)).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    if len(predictions) < 2:
        raise ValueError("At least two training samples are required for trust estimation.")

    lower, upper = float(np.min(predictions)), float(np.max(predictions))
    internal_edges = (
        np.array([], dtype="float32")
        if np.isclose(lower, upper)
        else np.linspace(lower, upper, bin_count + 1, dtype="float32")[1:-1]
    )
    bin_indices = np.searchsorted(internal_edges, predictions, side="right")
    actual_bin_count = len(internal_edges) + 1

    global_mse = float(np.mean(np.square(targets - predictions)))
    mse_values = np.full(actual_bin_count, global_mse, dtype="float32")

    for bin_index in range(actual_bin_count):
        mask = bin_indices == bin_index
        if np.any(mask):
            mse_values[bin_index] = np.mean(
                np.square(targets[mask] - predictions[mask])
            )

    trust_values = np.maximum(1.0 - mse_values / MAX_MSE, TRUST_EPSILON).astype("float32")

    return {
        "internal_edges": internal_edges,
        "trust_values": np.maximum(trust_values, TRUST_EPSILON),
        "mse_values": mse_values,
    }

def apply_trust_to_gate_probabilities(common_predictions, rare_predictions,
                                      gate_probabilities, common_profile, rare_profile):
    common_bins = np.searchsorted(common_profile["internal_edges"], common_predictions, side="right")
    rare_bins = np.searchsorted(rare_profile["internal_edges"], rare_predictions, side="right")
    trust = np.column_stack([common_profile["trust_values"][common_bins],
                             rare_profile["trust_values"][rare_bins]])
    adjusted = gate_probabilities * trust
    return adjusted / np.maximum(np.sum(adjusted, axis=1, keepdims=True), TRUST_EPSILON)


@keras.utils.register_keras_serializable(package="SEPTrust")
class PredictionBinTrust(layers.Layer):
    def __init__(self, internal_edges, trust_values, **kwargs):
        super().__init__(**kwargs)
        self.internal_edges_list = np.asarray(internal_edges, dtype="float32").tolist()
        self.trust_values_list = np.asarray(trust_values, dtype="float32").tolist()

    def call(self, predictions):
        edges = tf.constant(self.internal_edges_list, dtype=predictions.dtype)
        values = tf.constant(self.trust_values_list, dtype=predictions.dtype)
        indices = tf.searchsorted(edges, tf.reshape(predictions, [-1]), side="right")
        return tf.reshape(tf.gather(values, indices), [-1, 1])

    def get_config(self):
        config = super().get_config()
        config.update({"internal_edges": self.internal_edges_list,
                       "trust_values": self.trust_values_list})
        return config


@keras.utils.register_keras_serializable(package="SEPTrust")
class NormalizeTrustWeights(layers.Layer):
    def call(self, values):
        denominator = tf.maximum(
            tf.reduce_sum(values, axis=1, keepdims=True), TRUST_EPSILON
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



def build_trainable_gate_ensemble(
    common_expert,
    rare_expert,
    gate,
    common_trust_profile,
    rare_trust_profile,
):
    """Train the gate through the continuous trust-adjusted ensemble prediction."""
    common_expert.trainable = False
    rare_expert.trainable = False
    gate.trainable = True

    inputs = keras.Input(shape=common_expert.input_shape[1:], name="features")
    common_prediction = common_expert(inputs, training=False)
    rare_prediction = rare_expert(inputs, training=False)
    gate_trust = gate(inputs)

    common_quality = PredictionBinTrust(
        common_trust_profile["internal_edges"],
        common_trust_profile["trust_values"],
        name="common_prediction_bin_trust",
    )(common_prediction)
    rare_quality = PredictionBinTrust(
        rare_trust_profile["internal_edges"],
        rare_trust_profile["trust_values"],
        name="rare_prediction_bin_trust",
    )(rare_prediction)

    quality_values = layers.Concatenate(name="expert_trust_values")(
        [common_quality, rare_quality]
    )
    trust_quality = layers.Multiply(name="trust_weighted_gate")(
        [gate_trust, quality_values]
    )
    weights = NormalizeTrustWeights(name="normalized_trust_weighted_gate")(
        trust_quality
    )

    expert_predictions = layers.Concatenate(name="expert_predictions")(
        [common_prediction, rare_prediction]
    )
    final_prediction = layers.Dot(axes=1, name="gated_prediction")(
        [expert_predictions, weights]
    )

    model = keras.Model(
        inputs, final_prediction, name="trainable_trust_ensemble"
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"]
    )
    return model


class GateEnsembleAOREEarlyStopping(keras.callbacks.Callback):
    def __init__(self, x_val, y_val, region_labels, gate, patience=PATIENCE):
        super().__init__()
        self.x_val = x_val
        self.y_val = np.asarray(y_val).reshape(-1)
        self.rare = np.asarray(region_labels).reshape(-1).astype(bool)
        self.gate = gate
        self.patience = patience
        self.best_aore = np.inf
        self.best_epoch = 1
        self.best_gate_weights = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        pred = np.asarray(self.model(self.x_val, training=False)).reshape(-1)
        overall = mean_absolute_error(self.y_val, pred)
        rare = mean_absolute_error(self.y_val[self.rare], pred[self.rare])
        aore = (overall + rare) / 2.0

        if aore < self.best_aore:
            self.best_aore = aore
            self.best_epoch = epoch + 1
            self.best_gate_weights = self.gate.get_weights()
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_gate_weights is not None:
            self.gate.set_weights(self.best_gate_weights)


def evaluate_gate_alpha_with_kfold(
    features,
    targets,
    region_labels,
    common_expert,
    rare_expert,
    common_trust_profile,
    rare_trust_profile,
    alpha,
    fold_count=5,
    seed=42,
):
    labels = np.asarray(region_labels).reshape(-1).astype("int32")
    splitter = StratifiedKFold(
        n_splits=fold_count, shuffle=True, random_state=seed
    )
    aores, epochs = [], []

    for fold, (tr, va) in enumerate(splitter.split(features, labels), 1):
        tf.keras.backend.clear_session()
        set_global_determinism(seed + 100 + fold)
        gate = build_gate(features.shape[1])
        trainer = build_trainable_gate_ensemble(
            common_expert,
            rare_expert,
            gate,
            common_trust_profile,
            rare_trust_profile,
        )
        cb = GateEnsembleAOREEarlyStopping(
            features[va], targets[va], labels[va], gate
        )
        sw = reciprocal_weights(targets[tr], alpha)
        trainer.fit(
            features[tr],
            targets[tr],
            sample_weight=sw,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[cb],
            shuffle=False,
            verbose=0,
        )
        aores.append(cb.best_aore)
        epochs.append(cb.best_epoch)

    return float(np.mean(aores)), max(1, int(np.round(np.mean(epochs))))


def train_gate(
    features,
    targets,
    common_expert,
    rare_expert,
    common_trust_profile,
    rare_trust_profile,
    alpha,
    epoch_count,
    seed,
):
    tf.keras.backend.clear_session()
    set_global_determinism(seed)
    gate = build_gate(features.shape[1])
    trainer = build_trainable_gate_ensemble(
        common_expert,
        rare_expert,
        gate,
        common_trust_profile,
        rare_trust_profile,
    )
    trainer.fit(
        features,
        targets,
        sample_weight=reciprocal_weights(targets, alpha),
        epochs=epoch_count,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )
    return gate


def build_gated_ensemble(common_expert, rare_expert, gate,
                         common_trust_profile, rare_trust_profile):
    """Multiply gate probabilities by prediction-bin trust and renormalize."""
    common_expert.trainable = False
    rare_expert.trainable = False
    gate.trainable = False

    inputs = keras.Input(shape=common_expert.input_shape[1:], name="features")
    common_prediction = common_expert(inputs)
    rare_prediction = rare_expert(inputs)
    gate_probabilities = gate(inputs)
    common_trust = PredictionBinTrust(common_trust_profile["internal_edges"],
        common_trust_profile["trust_values"], name="common_prediction_bin_trust")(common_prediction)
    rare_trust = PredictionBinTrust(rare_trust_profile["internal_edges"],
        rare_trust_profile["trust_values"], name="rare_prediction_bin_trust")(rare_prediction)
    trust_values = layers.Concatenate(name="expert_trust_values")([common_trust, rare_trust])
    weighted_probabilities = layers.Multiply(name="trust_weighted_gate")(
        [gate_probabilities, trust_values])
    normalized_probabilities = NormalizeTrustWeights(
        name="normalized_trust_weighted_gate")(weighted_probabilities)

    expert_predictions = layers.Concatenate(name="expert_predictions")(
        [common_prediction, rare_prediction]
    )
    final_prediction = layers.Dot(axes=1, name="gated_prediction")(
        [expert_predictions, normalized_probabilities]
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
    trust = np.asarray(gate(features, training=False))
    labels = np.asarray(region_labels).reshape(-1).astype("int32")
    common, rare = labels == 0, labels == 1
    print(f"{split_name} mean g(C) for true common samples: {np.mean(trust[common, 0]):.4f}")
    print(f"{split_name} mean g(R) for true common samples: {np.mean(trust[common, 1]):.4f}")
    print(f"{split_name} mean g(C) for true rare samples: {np.mean(trust[rare, 0]):.4f}")
    print(f"{split_name} mean g(R) for true rare samples: {np.mean(trust[rare, 1]):.4f}")


def print_mse_trust_diagnostics(
    common_expert,
    rare_expert,
    gate,
    features,
    targets,
    region_labels,
    common_trust_profile,
    rare_trust_profile,
    split_name="Test",
):
    """Print per-instance gate/MSE-trust diagnostics and preference summaries."""
    common_predictions = np.asarray(
        common_expert(features, training=False)
    ).reshape(-1)
    rare_predictions = np.asarray(
        rare_expert(features, training=False)
    ).reshape(-1)
    gate_probabilities = np.asarray(gate(features, training=False))
    targets = np.asarray(targets).reshape(-1)
    labels = np.asarray(region_labels).reshape(-1).astype("int32")

    common_bins = np.searchsorted(
        common_trust_profile["internal_edges"], common_predictions, side="right"
    )
    rare_bins = np.searchsorted(
        rare_trust_profile["internal_edges"], rare_predictions, side="right"
    )

    common_mse = common_trust_profile["mse_values"][common_bins]
    rare_mse = rare_trust_profile["mse_values"][rare_bins]
    common_quality = common_trust_profile["trust_values"][common_bins]
    rare_quality = rare_trust_profile["trust_values"][rare_bins]

    raw_adjusted = gate_probabilities * np.column_stack(
        [common_quality, rare_quality]
    )
    adjusted_probabilities = raw_adjusted / np.maximum(
        np.sum(raw_adjusted, axis=1, keepdims=True), TRUST_EPSILON
    )
    final_predictions = (
        adjusted_probabilities[:, 0] * common_predictions
        + adjusted_probabilities[:, 1] * rare_predictions
    )

    print(f"\n{split_name} Per-Instance MSE Trust Diagnostics")
    print("=" * 176)
    print(
        f"{'Idx':>5} {'True y':>10} {'Type':>7} {'CommonPred':>11} {'RarePred':>11} "
        f"{'g(C)':>8} {'g(R)':>8} {'MSE(C)':>10} {'MSE(R)':>10} "
        f"{'Q(C)':>8} {'Q(R)':>8} {'Normalized(C)':>8} {'Normalized(R)':>8} {'FinalPred':>11}"
    )
    print("-" * 176)

    for index in range(len(targets)):
        sample_type = "RARE" if labels[index] == 1 else "COMMON"
        print(
            f"{index:5d} {targets[index]:10.4f} {sample_type:>7} "
            f"{common_predictions[index]:11.4f} {rare_predictions[index]:11.4f} "
            f"{gate_probabilities[index, 0]:8.4f} {gate_probabilities[index, 1]:8.4f} "
            f"{common_mse[index]:10.4f} {rare_mse[index]:10.4f} "
            f"{common_quality[index]:8.4f} {rare_quality[index]:8.4f} "
            f"{adjusted_probabilities[index, 0]:8.4f} "
            f"{adjusted_probabilities[index, 1]:8.4f} "
            f"{final_predictions[index]:11.4f}"
        )

    def print_preference_summary(mask, desired_index, group_name):
        count = int(np.sum(mask))
        if count == 0:
            print(f"\n{group_name}: no samples available.")
            return

        other_index = 1 - desired_index
        gate_prefers_desired = (
            gate_probabilities[mask, desired_index]
            > gate_probabilities[mask, other_index]
        )
        quality_matrix = np.column_stack([common_quality, rare_quality])
        quality_prefers_desired = (
            quality_matrix[mask, desired_index]
            > quality_matrix[mask, other_index]
        )

        both = gate_prefers_desired & quality_prefers_desired
        gate_only = gate_prefers_desired & ~quality_prefers_desired
        quality_only = ~gate_prefers_desired & quality_prefers_desired
        neither = ~gate_prefers_desired & ~quality_prefers_desired

        desired = "R" if desired_index == 1 else "C"
        other = "C" if desired_index == 1 else "R"
        print(f"\n{group_name} Preference Summary (n={count})")
        print("-" * 78)
        print(
            f"g({desired}) > g({other}) AND Q({desired}) > Q({other}): "
            f"{100.0 * np.mean(both):6.2f}% ({np.sum(both)}/{count})"
        )
        print(
            f"g({desired}) > g({other}) AND Q({desired}) <= Q({other}): "
            f"{100.0 * np.mean(gate_only):6.2f}% ({np.sum(gate_only)}/{count})"
        )
        print(
            f"g({desired}) <= g({other}) AND Q({desired}) > Q({other}): "
            f"{100.0 * np.mean(quality_only):6.2f}% ({np.sum(quality_only)}/{count})"
        )
        print(
            f"g({desired}) <= g({other}) AND Q({desired}) <= Q({other}): "
            f"{100.0 * np.mean(neither):6.2f}% ({np.sum(neither)}/{count})"
        )

    print_preference_summary(labels == 1, desired_index=1, group_name="Rare Samples")
    print_preference_summary(labels == 0, desired_index=0, group_name="Common Samples")

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

    if LOAD_SAVED_MODEL:
        required_model_paths = [COMMON_MODEL_PATH, RARE_MODEL_PATH, GATE_MODEL_PATH]
        missing_model_paths = [
            path for path in required_model_paths if not os.path.exists(path)
        ]
        if missing_model_paths:
            raise FileNotFoundError(
                "LOAD_SAVED_MODEL=True, but these saved model files are missing: "
                f"{missing_model_paths}. Run once with LOAD_SAVED_MODEL=False first."
            )
        print(f"\nLoading saved models from: {MODEL_DIRECTORY}")
        common_expert = keras.models.load_model(COMMON_MODEL_PATH)
        rare_expert = keras.models.load_model(RARE_MODEL_PATH)
        gate = keras.models.load_model(GATE_MODEL_PATH)
        print("Saved models loaded; skipping k-fold searches and retraining.")
    else:
        common_expert, common_alpha, common_epochs = select_and_train_expert(
            common_features, common_targets, "common_expert", seed + 1000
        )
        rare_expert, rare_alpha, rare_epochs = select_and_train_expert(
            rare_features, rare_targets, "rare_expert", seed + 2000
        )

    common_trust_profile = build_expert_trust_profile(
        common_expert, training_features, training_targets)
    rare_trust_profile = build_expert_trust_profile(
        rare_expert, training_features, training_targets)
    print(f"\nMSE trust mode: {TRUST_MODE}")
    print(f"Prediction bins per expert: {TRUST_BIN_COUNT}")

    if not LOAD_SAVED_MODEL:
        print("\nGate reciprocal-alpha search (trust gate trained through ensemble MSE)")
        best = (np.inf, None, None)
        for alpha in ALPHA_CANDIDATES:
            aore, epochs = evaluate_gate_alpha_with_kfold(
                training_features,
                training_targets,
                training_region_labels,
                common_expert,
                rare_expert,
                common_trust_profile,
                rare_trust_profile,
                alpha,
                fold_count=5,
                seed=seed,
            )
            print(
                f"  alpha={alpha}: mean validation AORE={aore:.4f}, "
                f"mean best epochs={epochs}"
            )
            if aore < best[0]:
                best = (aore, alpha, epochs)

        best_aore, best_gate_alpha, best_gate_epochs = best
        print(f"\nBest gate alpha: {best_gate_alpha}")
        print(f"Gate epochs: {best_gate_epochs}")
        print(f"Mean k-fold validation AORE: {best_aore:.4f}")

        gate = train_gate(
            training_features,
            training_targets,
            common_expert,
            rare_expert,
            common_trust_profile,
            rare_trust_profile,
            best_gate_alpha,
            best_gate_epochs,
            seed + 3000,
        )
        os.makedirs(MODEL_DIRECTORY, exist_ok=True)
        common_expert.save(COMMON_MODEL_PATH)
        rare_expert.save(RARE_MODEL_PATH)
        gate.save(GATE_MODEL_PATH)

    ensemble = build_gated_ensemble(common_expert, rare_expert, gate,
                                    common_trust_profile, rare_trust_profile)
    if not LOAD_SAVED_MODEL:
        ensemble.save(ENSEMBLE_MODEL_PATH)
        print(f"Saved models to: {MODEL_DIRECTORY}")



    print("\nGate Results")
    print_gate_summary(gate, training_features, training_region_labels, "Training")
    print_gate_summary(gate, testing_features, testing_region_labels, "Test")

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

    print_mse_trust_diagnostics(
        common_expert,
        rare_expert,
        gate,
        testing_features,
        testing_targets,
        testing_region_labels,
        common_trust_profile,
        rare_trust_profile,
        split_name="Test",
    )

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

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

    def on_epoch_end(self, epoch, logs=None):
        gate_probabilities = np.asarray(
            self.model(self.validation_features, training=False)
        )
        ensemble_predictions = (
            gate_probabilities[:, 0] * self.common_predictions
            + gate_probabilities[:, 1] * self.rare_predictions
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


def build_gated_ensemble(common_expert, rare_expert, gate):
    """Combine expert predictions using only the gate probabilities."""
    common_expert.trainable = False
    rare_expert.trainable = False
    gate.trainable = False

    inputs = keras.Input(shape=common_expert.input_shape[1:], name="features")
    common_prediction = common_expert(inputs)
    rare_prediction = rare_expert(inputs)
    gate_probabilities = gate(inputs)

    expert_predictions = layers.Concatenate(name="expert_predictions")(
        [common_prediction, rare_prediction]
    )
    final_prediction = layers.Dot(axes=1, name="gated_prediction")(
        [expert_predictions, gate_probabilities]
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
        common_features,
        common_targets,
        "common_expert",
        common_epochs,
        seed + 1000,
    )
    rare_expert = train_expert(
        rare_features,
        rare_targets,
        "rare_expert",
        rare_epochs,
        seed + 2000,
    )

    candidate_rare_class_weights = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
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

    # The experts are not retrained here. Only the winning gate is trained
    # once on the full training set using its selected class weight/epoch count.
    gate = train_gate(
        training_features,
        training_region_labels,
        rare_class_weight=best_gate_rare_class_weight,
        epoch_count=best_gate_epoch_count,
        seed=seed + 3000,
    )
    ensemble = build_gated_ensemble(common_expert, rare_expert, gate)

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

    ensemble.save("two_expert_gated_ensemble_sep_c.keras")
    common_expert.save("common_expert_sep_c.keras")
    rare_expert.save("rare_expert_sep_c.keras")
    gate.save("common_rare_gate_sep_c.keras")

    return ensemble, common_expert, rare_expert, gate


if __name__ == "__main__":
    final_ensemble, common_model, rare_model, gating_model = (
        run_two_expert_gated_ensemble(seed=42)
    )

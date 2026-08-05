import os
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import imbal

from imbal.regression import fit_kde, get_sample_densities, reciprocal_importance


def set_global_determinism(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


TARGET_COLUMN = "ln_peak_intensity"
TAIL_THRESHOLD = np.log(10)
TRAIN_DATA_PATH = "../../../tutorials/data/SEP-C/sep_10mev_training.csv"
TEST_DATA_PATH = "../../../tutorials/data/SEP-C/sep_10mev_testing.csv"
EXPECTED_INPUT_COUNT = 22


def load_sep_c_dataset(
    train_data_path=TRAIN_DATA_PATH,
    test_data_path=TEST_DATA_PATH,
    target_column=TARGET_COLUMN,
    tail_threshold=TAIL_THRESHOLD,
    expected_input_count=EXPECTED_INPUT_COUNT,
):
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    if target_column not in train_data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in the training data."
        )
    if target_column not in test_data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in the testing data."
        )

    training_features = train_data.drop(columns=[target_column]).values.astype(
        "float32"
    )
    testing_features = test_data.drop(columns=[target_column]).values.astype(
        "float32"
    )
    training_regression_targets = train_data[target_column].values.reshape(
        -1, 1
    ).astype("float32")
    testing_regression_targets = test_data[target_column].values.reshape(
        -1, 1
    ).astype("float32")

    if training_features.shape[1] != expected_input_count:
        raise ValueError(
            f"Expected {expected_input_count} input features, but found "
            f"{training_features.shape[1]} in the training data."
        )
    if testing_features.shape[1] != expected_input_count:
        raise ValueError(
            f"Expected {expected_input_count} input features, but found "
            f"{testing_features.shape[1]} in the testing data."
        )

    training_tail_labels = (
        training_regression_targets >= tail_threshold
    ).astype("float32")
    testing_tail_labels = (
        testing_regression_targets >= tail_threshold
    ).astype("float32")

    all_features = np.concatenate([training_features, testing_features], axis=0)
    all_regression_targets = np.concatenate(
        [training_regression_targets, testing_regression_targets],
        axis=0,
    )

    return (
        training_features,
        testing_features,
        training_regression_targets,
        testing_regression_targets,
        training_tail_labels,
        testing_tail_labels,
        all_features,
        all_regression_targets,
    )


def compute_aore(regression_targets, regression_predictions, tail_sample_mask):
    regression_targets = np.asarray(regression_targets).reshape(-1)
    regression_predictions = np.asarray(regression_predictions).reshape(-1)
    tail_sample_mask = np.asarray(tail_sample_mask).reshape(-1).astype(bool)

    if not np.any(tail_sample_mask):
        raise ValueError("No tail samples were found when computing AORE.")

    overall_mae = mean_absolute_error(
        regression_targets,
        regression_predictions,
    )
    tail_mae = mean_absolute_error(
        regression_targets[tail_sample_mask],
        regression_predictions[tail_sample_mask],
    )

    return (overall_mae + tail_mae) / 2


def build_encoder(prefix, input_count=EXPECTED_INPUT_COUNT):
    encoder_inputs = keras.Input(
        shape=(input_count,),
        name=f"{prefix}_input",
    )

    encoded = layers.Dense(
        18,
        activation="relu",
        name=f"{prefix}_dense_1",
    )(encoder_inputs)
    encoded = layers.Dense(
        12,
        activation="relu",
        name=f"{prefix}_dense_2",
    )(encoded)
    encoded = layers.Dense(
        8,
        activation="relu",
        name=f"{prefix}_dense_3",
    )(encoded)
    encoded = layers.Dense(
        6,
        activation="relu",
        name=f"{prefix}_dense_4",
    )(encoded)

    return keras.Model(
        encoder_inputs,
        encoded,
        name=f"{prefix}_encoder",
    )



def build_stage1_model(input_count):
    """Build the original-distribution representation model."""
    model_inputs = keras.Input(shape=(input_count,), name="features")
    encoder = build_encoder("stage1", input_count)
    latent = encoder(model_inputs)

    regression_output = layers.Dense(1, name="regression_output")(latent)
    classification_output = layers.Dense(
        1,
        activation="sigmoid",
        name="classification_output",
    )(latent)

    return keras.Model(
        model_inputs,
        {
            "regression_output": regression_output,
            "classification_output": classification_output,
        },
        name="stage1_original_distribution_model",
    )


def compile_stage1_model(model, classification_loss_weight=1.0):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "regression_output": "mse",
            "classification_output": "binary_crossentropy",
        },
        loss_weights={
            "regression_output": 1.0,
            "classification_output": classification_loss_weight,
        },
    )


def make_stage1_targets(regression_targets, tail_labels):
    return {
        "regression_output": regression_targets,
        "classification_output": tail_labels,
    }


def build_stage2_regressor(stage1_model, trainable_encoder=False):
    """Freeze the Stage 1 encoder and train a new weighted regression head."""
    encoder = stage1_model.get_layer("stage1_encoder")
    encoder.trainable = trainable_encoder

    model_inputs = keras.Input(
        shape=stage1_model.input_shape[1:],
        name="features",
    )
    latent = encoder(model_inputs)
    hidden = layers.Dense(
        6,
        activation="relu",
        name="stage2_regression_relu",
    )(latent)
    regression_output = layers.Dense(
        1,
        name="regression_output",
    )(hidden)

    return keras.Model(
        model_inputs,
        regression_output,
        name="stage2_reciprocal_weighted_regressor",
    )


def compile_stage2_regressor(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
def calculate_reciprocal_importance_weights(regression_targets, alpha=0.5):
    """Calculate reciprocal-importance weights from target-value density."""
    labels_for_kde = np.asarray(regression_targets).reshape(-1).copy()

    kde = fit_kde(labels_for_kde)
    sample_densities = get_sample_densities(labels_for_kde, kde)
    reciprocal_weights = reciprocal_importance(
        sample_densities,
        alpha=alpha,
    )

    reciprocal_weights = np.asarray(reciprocal_weights, dtype="float32")

    # Some imbal versions return weights inside an extra leading axis.
    if reciprocal_weights.ndim > 1:
        reciprocal_weights = reciprocal_weights[0]

    reciprocal_weights = reciprocal_weights.reshape(-1).astype("float32")

    if len(reciprocal_weights) != len(labels_for_kde):
        raise ValueError(
            "The reciprocal-weight array length does not match the number "
            "of regression targets."
        )

    return reciprocal_weights


class AOREEarlyStopping(keras.callbacks.Callback):
    def __init__(
        self,
        validation_features,
        validation_regression_targets,
        validation_tail_labels,
        prediction_output_name=None,
        patience=10,
    ):
        super().__init__()
        self.validation_features = validation_features
        self.validation_regression_targets = validation_regression_targets
        self.validation_tail_labels = validation_tail_labels
        self.prediction_output_name = prediction_output_name
        self.patience = patience

        self.best_aore = np.inf
        self.best_epoch = 1
        self.best_weights = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model(self.validation_features, training=False)

        if self.prediction_output_name is not None:
            predictions = predictions[self.prediction_output_name]

        current_aore = compute_aore(
            self.validation_regression_targets,
            np.asarray(predictions),
            self.validation_tail_labels,
        )

        if logs is not None:
            logs["val_aore"] = current_aore

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



def get_stage1_epoch_count_with_kfold(
    training_features,
    training_regression_targets,
    training_tail_labels,
    fold_count=5,
    seed=42,
):
    splitter = KFold(n_splits=fold_count, shuffle=True, random_state=seed)
    best_epochs = []

    for fold_training_indices, fold_validation_indices in splitter.split(
        training_features
    ):
        model = build_stage1_model(training_features.shape[1])
        compile_stage1_model(model)

        early_stopping = AOREEarlyStopping(
            validation_features=training_features[fold_validation_indices],
            validation_regression_targets=training_regression_targets[
                fold_validation_indices
            ],
            validation_tail_labels=training_tail_labels[
                fold_validation_indices
            ],
            prediction_output_name="regression_output",
            patience=10,
        )

        model.fit(
            training_features[fold_training_indices],
            make_stage1_targets(
                training_regression_targets[fold_training_indices],
                training_tail_labels[fold_training_indices],
            ),
            validation_data=(
                training_features[fold_validation_indices],
                make_stage1_targets(
                    training_regression_targets[fold_validation_indices],
                    training_tail_labels[fold_validation_indices],
                ),
            ),
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping],
            shuffle=False,
            verbose=0,
        )
        best_epochs.append(early_stopping.best_epoch)
        keras.backend.clear_session()

    return int(np.round(np.mean(best_epochs)))


def train_stage1_model(
    training_features,
    training_regression_targets,
    training_tail_labels,
    optimal_epoch_count,
):
    model = build_stage1_model(training_features.shape[1])
    compile_stage1_model(model)
    model.fit(
        training_features,
        make_stage1_targets(
            training_regression_targets,
            training_tail_labels,
        ),
        epochs=optimal_epoch_count,
        batch_size=32,
        shuffle=False,
        verbose=0,
    )
    return model


def get_stage2_epoch_count_with_kfold(
    stage1_model,
    training_features,
    training_regression_targets,
    training_tail_labels,
    reciprocal_alpha,
    fold_count=5,
    seed=42,
):
    splitter = KFold(n_splits=fold_count, shuffle=True, random_state=seed)
    best_epochs = []

    for fold_training_indices, fold_validation_indices in splitter.split(
        training_features
    ):
        model = build_stage2_regressor(stage1_model, trainable_encoder=False)
        compile_stage2_regressor(model)

        early_stopping = AOREEarlyStopping(
            validation_features=training_features[fold_validation_indices],
            validation_regression_targets=training_regression_targets[
                fold_validation_indices
            ],
            validation_tail_labels=training_tail_labels[
                fold_validation_indices
            ],
            patience=10,
        )

        model.fit(
            training_features[fold_training_indices],
            training_regression_targets[fold_training_indices],
            validation_data=(
                training_features[fold_validation_indices],
                training_regression_targets[fold_validation_indices],
            ),
            sample_weight=calculate_reciprocal_importance_weights(
                training_regression_targets[fold_training_indices],
                alpha=reciprocal_alpha,
            ),
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping],
            shuffle=False,
            verbose=0,
        )
        best_epochs.append(early_stopping.best_epoch)

    return int(np.round(np.mean(best_epochs)))


def train_stage2_regressor(
    stage1_model,
    training_features,
    training_regression_targets,
    training_tail_labels,
    reciprocal_alpha,
    optimal_epoch_count,
):
    model = build_stage2_regressor(stage1_model, trainable_encoder=False)
    compile_stage2_regressor(model)
    model.fit(
        training_features,
        training_regression_targets,
        sample_weight=calculate_reciprocal_importance_weights(
            training_regression_targets,
            alpha=reciprocal_alpha,
        ),
        epochs=optimal_epoch_count,
        batch_size=32,
        shuffle=False,
        verbose=0,
    )
    return model

def evaluate_regressor_with_aore_components(
    model,
    evaluation_features,
    evaluation_regression_targets,
    evaluation_tail_labels,
):
    predictions = model(evaluation_features, training=False).numpy().reshape(-1)
    targets = np.asarray(evaluation_regression_targets).reshape(-1)
    tail_mask = np.asarray(evaluation_tail_labels).reshape(-1).astype(bool)

    overall_mae = mean_absolute_error(targets, predictions)
    tail_mae = mean_absolute_error(targets[tail_mask], predictions[tail_mask])
    aore = (overall_mae + tail_mae) / 2

    return overall_mae, tail_mae, aore



def run_two_stage_decoupled_reciprocal_training(seed=42):
    set_global_determinism(seed)

    (
        training_features,
        testing_features,
        training_regression_targets,
        testing_regression_targets,
        training_tail_labels,
        testing_tail_labels,
        input_features,
        regression_targets,
    ) = load_sep_c_dataset()

    print(f"Training samples: {len(training_features)}")
    print(f"Testing samples: {len(testing_features)}")

    candidate_reciprocal_alphas = [
        0.05,
        0.1,
        0.15,
        0.20,
        0.25,
        0.3,
        0.4,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        2.5,
        5.0,
    ]

    # Stage 1: learn the representation from the original distribution.
    stage1_epoch_count = get_stage1_epoch_count_with_kfold(
        training_features,
        training_regression_targets,
        training_tail_labels,
        fold_count=5,
        seed=seed,
    )
    stage1_model = train_stage1_model(
        training_features,
        training_regression_targets,
        training_tail_labels,
        stage1_epoch_count,
    )

    stage1_training_predictions = stage1_model(
        training_features,
        training=False,
    )["regression_output"].numpy()
    stage1_training_aore = compute_aore(
        training_regression_targets,
        stage1_training_predictions,
        training_tail_labels,
    )

    print("\nStage 1: Original-Distribution Representation Learning")
    print(f"Optimal epoch count: {stage1_epoch_count}")
    print(f"Training AORE: {stage1_training_aore:.4f}")

    imbal.regression.tsne_visualization(
        stage1_model.get_layer("stage1_encoder"),
        input_features,
        regression_targets,
    )

    # Stage 2: freeze the Stage 1 encoder and scan reciprocal-weight alphas.
    best_stage2_model = None
    best_reciprocal_alpha = None
    best_stage2_epoch_count = None
    best_training_aore = np.inf

    for reciprocal_alpha in candidate_reciprocal_alphas:
        optimal_epoch_count = get_stage2_epoch_count_with_kfold(
            stage1_model,
            training_features,
            training_regression_targets,
            training_tail_labels,
            reciprocal_alpha=reciprocal_alpha,
            fold_count=5,
            seed=seed,
        )

        candidate_model = train_stage2_regressor(
            stage1_model,
            training_features,
            training_regression_targets,
            training_tail_labels,
            reciprocal_alpha=reciprocal_alpha,
            optimal_epoch_count=optimal_epoch_count,
        )

        _, _, candidate_training_aore = evaluate_regressor_with_aore_components(
            candidate_model,
            training_features,
            training_regression_targets,
            training_tail_labels,
        )

        print(
            "Stage 2 "
            f"reciprocal alpha={reciprocal_alpha}, "
            f"optimal epochs={optimal_epoch_count}, "
            f"training AORE={candidate_training_aore:.4f}"
        )

        if candidate_training_aore < best_training_aore:
            best_training_aore = candidate_training_aore
            best_stage2_model = candidate_model
            best_reciprocal_alpha = reciprocal_alpha
            best_stage2_epoch_count = optimal_epoch_count

    training_overall_mae, training_tail_mae, training_aore = (
        evaluate_regressor_with_aore_components(
            best_stage2_model,
            training_features,
            training_regression_targets,
            training_tail_labels,
        )
    )
    test_overall_mae, test_tail_mae, test_aore = (
        evaluate_regressor_with_aore_components(
            best_stage2_model,
            testing_features,
            testing_regression_targets,
            testing_tail_labels,
        )
    )

    print("\nBest Stage 2 Configuration")
    print(f"Reciprocal alpha: {best_reciprocal_alpha}")
    print(f"Optimal epoch count: {best_stage2_epoch_count}")
    print(f"Training overall MAE: {training_overall_mae:.4f}")
    print(f"Training rare MAE: {training_tail_mae:.4f}")
    print(f"Training AORE: {training_aore:.4f}")

    print("\nFinal Test Set Results")
    print(f"Test overall MAE: {test_overall_mae:.4f}")
    print(f"Test rare MAE: {test_tail_mae:.4f}")
    print(f"Test AORE: {test_aore:.4f}")

    imbal.regression.tsne_visualization(
        best_stage2_model,
        input_features,
        regression_targets,
    )

    predictions = best_stage2_model.predict(testing_features)
    imbal.regression.plot_true_vs_predictions(
        testing_regression_targets,
        predictions,
    )

    return best_stage2_model


if __name__ == "__main__":
    final_model = run_two_stage_decoupled_reciprocal_training(seed=42)

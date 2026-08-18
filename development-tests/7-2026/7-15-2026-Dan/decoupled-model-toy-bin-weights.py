import os
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import imbal


def set_global_determinism(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_toy_dataset(sample_count=500, seed=42, tail_quantile=0.95):
    """Create the same imbalanced toy regression dataset as the reference."""
    random_generator = np.random.default_rng(seed)

    feature_x1 = random_generator.normal(0, 1, sample_count)
    feature_x2 = random_generator.normal(0, 1, sample_count)
    distance_squared = feature_x1**2 + feature_x2**2

    input_features = np.stack(
        [
            np.stack([feature_x1, feature_x2], axis=1),
            np.stack(
                [distance_squared, np.zeros_like(distance_squared)],
                axis=1,
            ),
        ],
        axis=1,
    )

    input_features = input_features[..., np.newaxis].astype("float32")
    regression_targets = distance_squared.astype("float32")

    tail_threshold = np.quantile(distance_squared, tail_quantile)
    tail_labels = (distance_squared >= tail_threshold).astype("float32")

    return input_features, regression_targets, tail_labels, tail_threshold


def train_test_split_toy_dataset(
    input_features,
    regression_targets,
    tail_labels,
    training_fraction=0.8,
):
    training_sample_count = int(training_fraction * len(input_features))

    return (
        input_features[:training_sample_count],
        input_features[training_sample_count:],
        regression_targets[:training_sample_count],
        regression_targets[training_sample_count:],
        tail_labels[:training_sample_count],
        tail_labels[training_sample_count:],
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


def build_encoder(prefix):
    """Build the 2-D toy representation encoder used by Stage 1."""
    encoder_inputs = keras.Input(
        shape=(2, 2, 1),
        name=f"{prefix}_input",
    )

    encoded = layers.Flatten(name=f"{prefix}_flatten")(encoder_inputs)
    encoded = layers.Dense(
        16,
        activation="relu",
        name=f"{prefix}_dense_1",
    )(encoded)
    encoded = layers.Dense(
        12,
        activation="relu",
        name=f"{prefix}_dense_2",
    )(encoded)
    latent_space = layers.Dense(
        2,
        name=f"{prefix}_latent",
    )(encoded)

    return keras.Model(
        encoder_inputs,
        latent_space,
        name=f"{prefix}_encoder",
    )


def build_stage1_model():
    """Learn an unweighted representation from the original distribution."""
    model_inputs = keras.Input(shape=(2, 2, 1), name="model_input")
    encoder = build_encoder("stage1")
    latent_space = encoder(model_inputs)

    regression_output = layers.Dense(
        1,
        name="regression_output",
    )(latent_space)
    classification_output = layers.Dense(
        1,
        activation="sigmoid",
        name="classification_output",
    )(latent_space)

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
    """Freeze Stage 1 and train a new bin-weighted regression head."""
    encoder = stage1_model.get_layer("stage1_encoder")
    encoder.trainable = trainable_encoder

    model_inputs = keras.Input(shape=(2, 2, 1), name="model_input")
    latent_space = encoder(model_inputs)

    regression_hidden = layers.Dense(
        8,
        activation="relu",
        name="stage2_regression_relu",
    )(latent_space)
    regression_hidden = layers.Dense(
        4,
        activation="relu",
        name="stage2_regression_hidden",
    )(regression_hidden)
    regression_output = layers.Dense(
        1,
        name="regression_output",
    )(regression_hidden)

    return keras.Model(
        model_inputs,
        regression_output,
        name="stage2_bin_weighted_regressor",
    )


def compile_stage2_regressor(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )


def create_regression_sample_weights(tail_labels, tail_weight):
    tail_labels = np.asarray(tail_labels).reshape(-1)
    sample_weights = np.ones_like(tail_labels, dtype="float32")
    sample_weights[tail_labels == 1] = tail_weight
    return sample_weights


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
    splitter = KFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=seed,
    )
    best_epochs = []

    for fold_training_indices, fold_validation_indices in splitter.split(
        training_features
    ):
        model = build_stage1_model()
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
    model = build_stage1_model()
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
    tail_weight,
    fold_count=5,
    seed=42,
):
    splitter = KFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=seed,
    )
    best_epochs = []

    for fold_training_indices, fold_validation_indices in splitter.split(
        training_features
    ):
        model = build_stage2_regressor(
            stage1_model,
            trainable_encoder=False,
        )
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
            sample_weight=create_regression_sample_weights(
                training_tail_labels[fold_training_indices],
                tail_weight,
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
    tail_weight,
    optimal_epoch_count,
):
    model = build_stage2_regressor(
        stage1_model,
        trainable_encoder=False,
    )
    compile_stage2_regressor(model)

    model.fit(
        training_features,
        training_regression_targets,
        sample_weight=create_regression_sample_weights(
            training_tail_labels,
            tail_weight,
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

    if not np.any(tail_mask):
        raise ValueError("No tail samples were found during evaluation.")

    overall_mae = mean_absolute_error(targets, predictions)
    tail_mae = mean_absolute_error(targets[tail_mask], predictions[tail_mask])
    aore = (overall_mae + tail_mae) / 2

    return overall_mae, tail_mae, aore


def plot_stage1_latent_space(
    stage1_model,
    input_features,
    regression_targets,
):
    encoder = stage1_model.get_layer("stage1_encoder")
    latent_representations = encoder(
        input_features,
        training=False,
    ).numpy()

    plt.figure()
    scatter_plot = plt.scatter(
        latent_representations[:, 0],
        latent_representations[:, 1],
        c=np.asarray(regression_targets).reshape(-1),
        cmap="viridis",
    )
    plt.colorbar(scatter_plot, label="t = x1² + x2²")
    plt.xlabel("Representation Dimension 1")
    plt.ylabel("Representation Dimension 2")
    plt.title("Stage 1 Original-Distribution Representation")
    plt.show()


def run_two_stage_decoupled_bin_weighted_training(seed=42):
    set_global_determinism(seed)

    (
        input_features,
        regression_targets,
        tail_labels,
        tail_threshold,
    ) = make_toy_dataset(
        sample_count=500,
        seed=seed,
        tail_quantile=0.95,
    )

    (
        training_features,
        testing_features,
        training_regression_targets,
        testing_regression_targets,
        training_tail_labels,
        testing_tail_labels,
    ) = train_test_split_toy_dataset(
        input_features,
        regression_targets,
        tail_labels,
        training_fraction=0.8,
    )

    print(f"Tail threshold: {tail_threshold:.4f}")
    print(f"Training samples: {len(training_features)}")
    print(f"Testing samples: {len(testing_features)}")
    print(f"Training tail samples: {int(np.sum(training_tail_labels))}")
    print(f"Testing tail samples: {int(np.sum(testing_tail_labels))}")

    candidate_tail_weights = [
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

    # -------------------------------------------------------------
    # Stage 1: Learn an unweighted original-distribution representation.
    # -------------------------------------------------------------
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

    stage1_predictions = stage1_model(
        training_features,
        training=False,
    )["regression_output"].numpy()
    stage1_training_aore = compute_aore(
        training_regression_targets,
        stage1_predictions,
        training_tail_labels,
    )

    print("\nStage 1: Original-Distribution Representation Learning")
    print(f"Optimal epoch count: {stage1_epoch_count}")
    print(f"Training AORE: {stage1_training_aore:.4f}")

    plot_stage1_latent_space(
        stage1_model,
        input_features,
        regression_targets,
    )

    # -------------------------------------------------------------
    # Stage 2: Freeze Stage 1 and scan explicit tail-bin weights.
    # -------------------------------------------------------------
    best_stage2_model = None
    best_tail_weight = None
    best_stage2_epoch_count = None
    best_training_aore = np.inf

    for tail_weight in candidate_tail_weights:
        optimal_epoch_count = get_stage2_epoch_count_with_kfold(
            stage1_model,
            training_features,
            training_regression_targets,
            training_tail_labels,
            tail_weight=tail_weight,
            fold_count=5,
            seed=seed,
        )

        candidate_model = train_stage2_regressor(
            stage1_model,
            training_features,
            training_regression_targets,
            training_tail_labels,
            tail_weight=tail_weight,
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
            f"tail weight={tail_weight}, "
            f"optimal epochs={optimal_epoch_count}, "
            f"training AORE={candidate_training_aore:.4f}"
        )

        if candidate_training_aore < best_training_aore:
            best_training_aore = candidate_training_aore
            best_stage2_model = candidate_model
            best_tail_weight = tail_weight
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
    print(f"Tail weight: {best_tail_weight}")
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
    final_model = run_two_stage_decoupled_bin_weighted_training(seed=42)

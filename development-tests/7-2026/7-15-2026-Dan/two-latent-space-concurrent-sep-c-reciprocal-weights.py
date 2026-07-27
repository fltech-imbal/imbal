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

from imbal.regression import fit_kde, get_sample_densities, reciprocal_importance
import imbal


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
    expected_input_count=EXPECTED_INPUT_COUNT
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
        -1,
        1
    ).astype("float32")

    testing_regression_targets = test_data[target_column].values.reshape(
        -1,
        1
    ).astype("float32")

    if training_features.shape[1] != expected_input_count:
        raise ValueError(
            "Expected "
            f"{expected_input_count} input features, but found "
            f"{training_features.shape[1]} in the training data."
        )

    if testing_features.shape[1] != expected_input_count:
        raise ValueError(
            "Expected "
            f"{expected_input_count} input features, but found "
            f"{testing_features.shape[1]} in the testing data."
        )

    training_tail_labels = (
        training_regression_targets >= tail_threshold
    ).astype("float32")

    testing_tail_labels = (
        testing_regression_targets >= tail_threshold
    ).astype("float32")

    all_features = np.concatenate(
        [training_features, testing_features],
        axis=0
    )

    all_regression_targets = np.concatenate(
        [training_regression_targets, testing_regression_targets],
        axis=0
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
        raise ValueError(
            "No tail samples were found when computing AORE. Check the "
            "tail threshold or the split being evaluated."
        )

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

    encoded = encoder_inputs
    encoded = layers.Dense(
        18,
        activation="relu",
        name=f"{prefix}_dense_1",
    )(encoded)
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


def build_concurrent_representation_model(
    input_count=EXPECTED_INPUT_COUNT,
):
    """
    Learn two independent latent spaces concurrently from the same raw input.

    The original encoder is optimized using uniform sample weights. The
    density-aware encoder is optimized using reciprocal-importance weights.
    Separate auxiliary outputs allow the two branches to receive different
    sample weights during the same model.fit call.
    """
    model_inputs = keras.Input(
        shape=(input_count,),
        name="features",
    )

    original_encoder = build_encoder("original", input_count)
    density_encoder = build_encoder("density", input_count)

    original_latent = original_encoder(model_inputs)
    density_latent = density_encoder(model_inputs)

    original_regression_output = layers.Dense(
        1,
        name="original_regression_output",
    )(original_latent)
    original_classification_output = layers.Dense(
        1,
        activation="sigmoid",
        name="original_classification_output",
    )(original_latent)

    density_regression_output = layers.Dense(
        1,
        name="density_regression_output",
    )(density_latent)
    density_classification_output = layers.Dense(
        1,
        activation="sigmoid",
        name="density_classification_output",
    )(density_latent)

    return keras.Model(
        inputs=model_inputs,
        outputs={
            "original_regression_output": original_regression_output,
            "original_classification_output": original_classification_output,
            "density_regression_output": density_regression_output,
            "density_classification_output": density_classification_output,
        },
        name="concurrent_reciprocal_representation_model",
    )


def compile_concurrent_representation_model(
    model,
    classification_loss_weight=1.0,
):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "original_regression_output": "mse",
            "original_classification_output": "binary_crossentropy",
            "density_regression_output": "mse",
            "density_classification_output": "binary_crossentropy",
        },
        loss_weights={
            "original_regression_output": 1.0,
            "original_classification_output": classification_loss_weight,
            "density_regression_output": 1.0,
            "density_classification_output": classification_loss_weight,
        },
    )


def make_concurrent_targets(regression_targets, tail_labels):
    return {
        "original_regression_output": regression_targets,
        "original_classification_output": tail_labels,
        "density_regression_output": regression_targets,
        "density_classification_output": tail_labels,
    }


def calculate_reciprocal_importance_weights(regression_targets, alpha=0.5):
    labels_for_kde = np.asarray(regression_targets).reshape(-1).copy()

    kde = fit_kde(labels_for_kde)
    sample_densities = get_sample_densities(labels_for_kde, kde)
    reciprocal_weights = reciprocal_importance(
        sample_densities,
        alpha=alpha,
    )

    reciprocal_weights = np.asarray(reciprocal_weights, dtype="float32")

    if reciprocal_weights.ndim > 1:
        reciprocal_weights = reciprocal_weights[0]

    return reciprocal_weights.reshape(-1).astype("float32")


def create_concurrent_sample_weights(regression_targets, reciprocal_alpha):
    reciprocal_weights = calculate_reciprocal_importance_weights(
        regression_targets,
        alpha=reciprocal_alpha,
    )
    uniform_weights = np.ones_like(reciprocal_weights, dtype="float32")

    return {
        "original_regression_output": uniform_weights,
        "original_classification_output": uniform_weights,
        "density_regression_output": reciprocal_weights,
        "density_classification_output": reciprocal_weights,
    }


def build_fused_regressor(
    concurrent_representation_model,
    trainable_encoders=False,
):
    """
    Combine the two learned latent spaces and train one downstream regressor.

    Concatenation supplies both representations to the regressor, whose
    learned weights determine which latent features are most useful.
    """
    original_encoder = concurrent_representation_model.get_layer(
        "original_encoder"
    )
    density_encoder = concurrent_representation_model.get_layer(
        "density_encoder"
    )

    original_encoder.trainable = trainable_encoders
    density_encoder.trainable = trainable_encoders

    model_inputs = keras.Input(
        shape=concurrent_representation_model.input_shape[1:],
        name="features",
    )
    original_latent = original_encoder(model_inputs)
    density_latent = density_encoder(model_inputs)

    fused_latent = layers.Concatenate(name="fused_latent")([
        original_latent,
        density_latent,
    ])

    fused_hidden = layers.Dense(
        6,
        activation="relu",
        name="fused_regression_relu",
    )(fused_latent)
    regression_output = layers.Dense(
        1,
        name="regression_output",
    )(fused_hidden)

    return keras.Model(
        model_inputs,
        regression_output,
        name="fused_reciprocal_regressor",
    )


def compile_fused_regressor(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )


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


def get_concurrent_representation_epoch_count_with_kfold(
    training_features,
    training_regression_targets,
    training_tail_labels,
    reciprocal_alpha,
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
        fold_model = build_concurrent_representation_model(
            input_count=training_features.shape[1]
        )
        compile_concurrent_representation_model(fold_model)

        fold_training_targets = make_concurrent_targets(
            training_regression_targets[fold_training_indices],
            training_tail_labels[fold_training_indices],
        )
        fold_validation_targets = make_concurrent_targets(
            training_regression_targets[fold_validation_indices],
            training_tail_labels[fold_validation_indices],
        )
        fold_sample_weights = create_concurrent_sample_weights(
            training_regression_targets[fold_training_indices],
            reciprocal_alpha=reciprocal_alpha,
        )

        early_stopping = AOREEarlyStopping(
            validation_features=training_features[fold_validation_indices],
            validation_regression_targets=training_regression_targets[
                fold_validation_indices
            ],
            validation_tail_labels=training_tail_labels[
                fold_validation_indices
            ],
            prediction_output_name="density_regression_output",
            patience=10,
        )

        fold_model.fit(
            training_features[fold_training_indices],
            fold_training_targets,
            validation_data=(
                training_features[fold_validation_indices],
                fold_validation_targets,
            ),
            sample_weight=fold_sample_weights,
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping],
            shuffle=False,
            verbose=0,
        )

        best_epochs.append(early_stopping.best_epoch)

    return int(np.round(np.mean(best_epochs)))


def train_concurrent_representation_model(
    training_features,
    training_regression_targets,
    training_tail_labels,
    reciprocal_alpha,
    optimal_epoch_count,
):
    model = build_concurrent_representation_model(
        input_count=training_features.shape[1]
    )
    compile_concurrent_representation_model(model)

    model.fit(
        training_features,
        make_concurrent_targets(
            training_regression_targets,
            training_tail_labels,
        ),
        sample_weight=create_concurrent_sample_weights(
            training_regression_targets,
            reciprocal_alpha=reciprocal_alpha,
        ),
        epochs=optimal_epoch_count,
        batch_size=32,
        shuffle=False,
        verbose=0,
    )

    return model


def get_fused_regressor_epoch_count_with_kfold(
    concurrent_representation_model,
    training_features,
    training_regression_targets,
    training_tail_labels,
    reciprocal_alpha,
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
        fold_model = build_fused_regressor(
            concurrent_representation_model,
            trainable_encoders=False,
        )
        compile_fused_regressor(fold_model)

        fold_sample_weights = calculate_reciprocal_importance_weights(
            training_regression_targets[fold_training_indices],
            alpha=reciprocal_alpha,
        )

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

        fold_model.fit(
            training_features[fold_training_indices],
            training_regression_targets[fold_training_indices],
            validation_data=(
                training_features[fold_validation_indices],
                training_regression_targets[fold_validation_indices],
            ),
            sample_weight=fold_sample_weights,
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping],
            shuffle=False,
            verbose=0,
        )

        best_epochs.append(early_stopping.best_epoch)

    return int(np.round(np.mean(best_epochs)))


def train_fused_regressor(
    concurrent_representation_model,
    training_features,
    training_regression_targets,
    reciprocal_alpha,
    optimal_epoch_count,
):
    model = build_fused_regressor(
        concurrent_representation_model,
        trainable_encoders=False,
    )
    compile_fused_regressor(model)

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

    if not np.any(tail_mask):
        raise ValueError(
            "No tail samples were found when computing AORE components."
        )

    overall_mae = mean_absolute_error(targets, predictions)
    tail_mae = mean_absolute_error(targets[tail_mask], predictions[tail_mask])
    aore = (overall_mae + tail_mae) / 2

    return overall_mae, tail_mae, aore


def plot_latent_space(
    concurrent_representation_model,
    encoder_name,
    input_features,
    regression_targets,
    title,
):
    encoder = concurrent_representation_model.get_layer(encoder_name)
    latent_representations = encoder(
        input_features,
        training=False,
    ).numpy()

    plt.figure()
    scatter_plot = plt.scatter(
        latent_representations[:, 0],
        latent_representations[:, 1],
        c=regression_targets.reshape(-1),
        cmap="viridis",
    )
    plt.colorbar(scatter_plot, label=TARGET_COLUMN)
    plt.xlabel("Representation Dimension 1")
    plt.ylabel("Representation Dimension 2")
    plt.title(title)
    plt.show()


def run_concurrent_two_latent_sep_reciprocal_training(seed=42):
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

    # -------------------------------------------------------------
    # Phase 1: Learn both latent spaces concurrently.
    # -------------------------------------------------------------
    best_representation_model = None
    best_representation_alpha = None
    best_representation_epoch_count = None
    best_density_branch_training_aore = np.inf

    for candidate_alpha in candidate_reciprocal_alphas:
        optimal_epoch_count = (
            get_concurrent_representation_epoch_count_with_kfold(
                training_features,
                training_regression_targets,
                training_tail_labels,
                reciprocal_alpha=candidate_alpha,
                fold_count=5,
                seed=seed,
            )
        )

        candidate_model = train_concurrent_representation_model(
            training_features,
            training_regression_targets,
            training_tail_labels,
            reciprocal_alpha=candidate_alpha,
            optimal_epoch_count=optimal_epoch_count,
        )

        density_predictions = candidate_model(
            training_features,
            training=False,
        )["density_regression_output"].numpy()

        candidate_aore = compute_aore(
            training_regression_targets,
            density_predictions,
            training_tail_labels,
        )

        print(
            "Concurrent representation learning "
            f"reciprocal alpha={candidate_alpha}, "
            f"optimal epochs={optimal_epoch_count}, "
            f" density-branch training AORE={candidate_aore:.4f}"
        )

        if candidate_aore < best_density_branch_training_aore:
            best_density_branch_training_aore = candidate_aore
            best_representation_model = candidate_model
            best_representation_alpha = candidate_alpha
            best_representation_epoch_count = optimal_epoch_count

    print("\nBest Concurrent Representation Configuration")
    print(f"Reciprocal alpha: {best_representation_alpha}")
    print(f"Optimal epoch count: {best_representation_epoch_count}")
    print(
        "Density branch training AORE: "
        f"{best_density_branch_training_aore:.4f}"
    )

    original_encoder = best_representation_model.get_layer(
        "original_encoder"
    )
    density_encoder = best_representation_model.get_layer(
        "density_encoder"
    )

    imbal.regression.tsne_visualization(
        original_encoder,
        input_features,
        regression_targets,
    )

    imbal.regression.tsne_visualization(
        density_encoder,
        input_features,
        regression_targets,
    )

    # -------------------------------------------------------------
    # Phase 2: Freeze both encoders and train a fused regressor.
    # -------------------------------------------------------------
    best_fused_regressor = None
    best_regressor_alpha = None
    best_regressor_epoch_count = None
    best_training_aore = np.inf

    for candidate_alpha in candidate_reciprocal_alphas:
        optimal_epoch_count = get_fused_regressor_epoch_count_with_kfold(
            best_representation_model,
            training_features,
            training_regression_targets,
            training_tail_labels,
            reciprocal_alpha=candidate_alpha,
            fold_count=5,
            seed=seed,
        )

        candidate_regressor = train_fused_regressor(
            best_representation_model,
            training_features,
            training_regression_targets,
            reciprocal_alpha=candidate_alpha,
            optimal_epoch_count=optimal_epoch_count,
        )

        _, _, candidate_training_aore = evaluate_regressor_with_aore_components(
            candidate_regressor,
            training_features,
            training_regression_targets,
            training_tail_labels,
        )

        print(
            "Fused regressor "
            f"reciprocal alpha={candidate_alpha}, "
            f"optimal epochs={optimal_epoch_count}, "
            f"training AORE={candidate_training_aore:.4f}"
        )

        if candidate_training_aore < best_training_aore:
            best_training_aore = candidate_training_aore
            best_fused_regressor = candidate_regressor
            best_regressor_alpha = candidate_alpha
            best_regressor_epoch_count = optimal_epoch_count

    training_overall_mae, training_tail_mae, training_aore = (
        evaluate_regressor_with_aore_components(
            best_fused_regressor,
            training_features,
            training_regression_targets,
            training_tail_labels,
        )
    )
    test_overall_mae, test_tail_mae, test_aore = (
        evaluate_regressor_with_aore_components(
            best_fused_regressor,
            testing_features,
            testing_regression_targets,
            testing_tail_labels,
        )
    )

    print("\nBest Fused Regressor Configuration")
    print(f"Reciprocal alpha: {best_regressor_alpha}")
    print(f"Optimal epoch count: {best_regressor_epoch_count}")
    print(f"Training overall MAE: {training_overall_mae:.4f}")
    print(f"Training rare MAE: {training_tail_mae:.4f}")
    print(f"Training AORE: {training_aore:.4f}")

    print("\nFinal Test Set Results")
    print(f"Representation reciprocal alpha: {best_representation_alpha}")
    print(f"Regressor reciprocal alpha: {best_regressor_alpha}")
    print(f"Test overall MAE: {test_overall_mae:.4f}")
    print(f"Test rare MAE: {test_tail_mae:.4f}")
    print(f"Test AORE: {test_aore:.4f}")

    imbal.regression.tsne_visualization(best_fused_regressor, input_features, regression_targets)

    predictions = best_fused_regressor.predict(testing_features)

    imbal.regression.plot_true_vs_predictions(testing_regression_targets, predictions)

    return best_fused_regressor


if __name__ == "__main__":
    final_model = run_concurrent_two_latent_sep_reciprocal_training(seed=42)

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
    regression_targets = regression_targets.reshape(-1)
    regression_predictions = regression_predictions.reshape(-1)
    tail_sample_mask = np.asarray(tail_sample_mask).reshape(-1).astype(bool)

    overall_mean_absolute_error = mean_absolute_error(
        regression_targets,
        regression_predictions
    )

    if not np.any(tail_sample_mask):
        raise ValueError(
            "No tail samples were found when computing AORE. Check the "
            "tail threshold or the split being evaluated."
        )

    tail_mean_absolute_error = mean_absolute_error(
        regression_targets[tail_sample_mask],
        regression_predictions[tail_sample_mask]
    )

    return (overall_mean_absolute_error + tail_mean_absolute_error) / 2


def build_representation_learning_model(input_count=EXPECTED_INPUT_COUNT):
    model_inputs = keras.Input(shape=(input_count,), name="features")

    feature_layer_1 = layers.Dense(
        18,
        activation="relu",
        name="feature_dense_1"
    )(model_inputs)

    feature_layer_2 = layers.Dense(
        12,
        activation="relu",
        name="feature_dense_2"
    )(feature_layer_1)

    learned_features = layers.Dense(
        2,
        name="feature_dense_3"
    )(feature_layer_2)

    regression_output = layers.Dense(
        1,
        name="regression_output"
    )(learned_features)

    classification_output = layers.Dense(
        1,
        activation="sigmoid",
        name="classification_output"
    )(learned_features)

    return keras.Model(
        inputs=model_inputs,
        outputs={
            "regression_output": regression_output,
            "classification_output": classification_output,
        }
    )


def get_feature_extractor_output(previous_model):
    try:
        return previous_model.get_layer("feature_dense_3").output
    except ValueError:
        try:
            return previous_model.get_layer("new_regression_hidden").output
        except ValueError:
            nested_feature_extractor = previous_model.get_layer(
                "feature_extractor"
            )
            return nested_feature_extractor(previous_model.input)


def create_retraining_model_with_new_heads(
    previous_model,
    trainable_feature_extractor=False
):
    feature_extractor_output = get_feature_extractor_output(previous_model)

    feature_extractor = keras.Model(
        previous_model.input,
        feature_extractor_output,
        name="feature_extractor"
    )

    feature_extractor.trainable = trainable_feature_extractor

    model_inputs = keras.Input(
        shape=previous_model.input_shape[1:],
        name="features"
    )

    extracted_features = feature_extractor(model_inputs)

    regression_relu_layer = layers.Dense(
        8,
        activation="relu",
        name="new_regression_relu_layer"
    )(extracted_features)

    regression_hidden_layer = layers.Dense(
        2,
        name="new_regression_hidden"
    )(regression_relu_layer)

    regression_output = layers.Dense(
        1,
        name="regression_output"
    )(regression_hidden_layer)

    classification_output = layers.Dense(
        1,
        activation="sigmoid",
        name="classification_output"
    )(extracted_features)

    return keras.Model(
        inputs=model_inputs,
        outputs={
            "regression_output": regression_output,
            "classification_output": classification_output,
        }
    )


def compile_training_model(model, classification_loss_weight=1.0):
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
        metrics={
            "regression_output": ["mae"],
            "classification_output": ["accuracy"],
        }
    )


def create_tail_reweighted_sample_weights(tail_labels, tail_weight):
    tail_labels = np.asarray(tail_labels).reshape(-1)

    regression_sample_weights = np.ones_like(tail_labels, dtype="float32")
    classification_sample_weights = np.ones_like(tail_labels, dtype="float32")

    regression_sample_weights[tail_labels == 1] = tail_weight
    classification_sample_weights[tail_labels == 1] = tail_weight

    return {
        "regression_output": regression_sample_weights,
        "classification_output": classification_sample_weights,
    }


class AOREEarlyStopping(keras.callbacks.Callback):
    def __init__(
        self,
        validation_features,
        validation_regression_targets,
        validation_tail_labels,
        patience=10
    ):
        super().__init__()

        self.validation_features = validation_features
        self.validation_regression_targets = validation_regression_targets
        self.validation_tail_labels = validation_tail_labels
        self.patience = patience

        self.best_aore = np.inf
        self.best_epoch = 1
        self.best_weights = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        model_predictions = self.model(
            self.validation_features,
            training=False
        )

        regression_predictions = model_predictions["regression_output"].numpy()

        tail_sample_mask = np.asarray(
            self.validation_tail_labels
        ).reshape(-1).astype(bool)

        current_aore = compute_aore(
            self.validation_regression_targets,
            regression_predictions,
            tail_sample_mask
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


def get_optimal_epoch_count_with_kfold(
    build_model_function,
    training_features,
    training_regression_targets,
    training_tail_labels,
    tail_weight=1.0,
    fold_count=5,
    seed=42
):
    kfold_splitter = KFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=seed
    )

    optimal_epoch_counts = []

    for fold_training_indices, fold_validation_indices in kfold_splitter.split(
        training_features
    ):
        fold_model = build_model_function()
        compile_training_model(fold_model)

        fold_sample_weights = create_tail_reweighted_sample_weights(
            training_tail_labels[fold_training_indices],
            tail_weight
        )

        aore_early_stopping_callback = AOREEarlyStopping(
            validation_features=training_features[fold_validation_indices],
            validation_regression_targets=training_regression_targets[
                fold_validation_indices
            ],
            validation_tail_labels=training_tail_labels[
                fold_validation_indices
            ],
            patience=10
        )

        fold_model.fit(
            training_features[fold_training_indices],
            {
                "regression_output": training_regression_targets[
                    fold_training_indices
                ],
                "classification_output": training_tail_labels[
                    fold_training_indices
                ],
            },
            validation_data=(
                training_features[fold_validation_indices],
                {
                    "regression_output": training_regression_targets[
                        fold_validation_indices
                    ],
                    "classification_output": training_tail_labels[
                        fold_validation_indices
                    ],
                }
            ),
            sample_weight=fold_sample_weights,
            epochs=200,
            batch_size=32,
            callbacks=[aore_early_stopping_callback],
            shuffle=False,
            verbose=0
        )

        optimal_epoch_counts.append(aore_early_stopping_callback.best_epoch)

    return int(np.round(np.mean(optimal_epoch_counts)))


def retrain_for_optimal_epoch_count(
    model,
    training_features,
    training_regression_targets,
    training_tail_labels,
    optimal_epoch_count,
    tail_weight=1.0
):
    training_sample_weights = create_tail_reweighted_sample_weights(
        training_tail_labels,
        tail_weight
    )

    model.fit(
        training_features,
        {
            "regression_output": training_regression_targets,
            "classification_output": training_tail_labels,
        },
        sample_weight=training_sample_weights,
        epochs=optimal_epoch_count,
        batch_size=32,
        shuffle=False,
        verbose=0
    )

    return model


def evaluate_model_with_aore(
    model,
    evaluation_features,
    evaluation_regression_targets,
    evaluation_tail_labels
):
    model_predictions = model(
        evaluation_features,
        training=False
    )

    regression_predictions = model_predictions["regression_output"].numpy()

    tail_sample_mask = np.asarray(
        evaluation_tail_labels
    ).reshape(-1).astype(bool)

    return compute_aore(
        evaluation_regression_targets,
        regression_predictions,
        tail_sample_mask
    )


def evaluate_model_with_aore_components(
    model,
    evaluation_features,
    evaluation_regression_targets,
    evaluation_tail_labels
):
    model_predictions = model(
        evaluation_features,
        training=False
    )

    regression_predictions = model_predictions[
        "regression_output"
    ].numpy().reshape(-1)

    evaluation_regression_targets = evaluation_regression_targets.reshape(-1)

    tail_sample_mask = np.asarray(
        evaluation_tail_labels
    ).reshape(-1).astype(bool)

    if not np.any(tail_sample_mask):
        raise ValueError(
            "No tail samples were found when computing AORE components. "
            "Check the tail threshold or the split being evaluated."
        )

    overall_mae = mean_absolute_error(
        evaluation_regression_targets,
        regression_predictions
    )

    tail_mae = mean_absolute_error(
        evaluation_regression_targets[tail_sample_mask],
        regression_predictions[tail_sample_mask]
    )

    aore = (overall_mae + tail_mae) / 2

    return overall_mae, tail_mae, aore


def plot_representation_from_output(
    model,
    latent_output,
    input_features,
    regression_targets,
    title
):
    latent_space_model = keras.Model(
        inputs=model.input,
        outputs=latent_output
    )

    latent_representations = latent_space_model(
        input_features,
        training=False
    ).numpy()

    plt.figure()
    scatter_plot = plt.scatter(
        latent_representations[:, 0],
        latent_representations[:, 1],
        c=regression_targets.reshape(-1),
        cmap="viridis"
    )

    plt.colorbar(scatter_plot, label=TARGET_COLUMN)
    plt.xlabel("Representation Dimension 1")
    plt.ylabel("Representation Dimension 2")
    plt.title(title)
    plt.show()


def plot_stage1_representation_space(
    stage1_model,
    input_features,
    regression_targets
):
    latent_output = stage1_model.get_layer("feature_dense_3").output

    plot_representation_from_output(
        stage1_model,
        latent_output,
        input_features,
        regression_targets,
        title="Stage 1 Learned 2D Representation Space"
    )


def plot_stage2_representation_space(
    stage2_model,
    input_features,
    regression_targets
):
    latent_output = stage2_model.get_layer("new_regression_hidden").output

    plot_representation_from_output(
        stage2_model,
        latent_output,
        input_features,
        regression_targets,
        title="Stage 2 Best Density-Aware Representation Space"
    )


def plot_stage3_representation_space(
    stage3_model,
    input_features,
    regression_targets
):
    latent_output = stage3_model.get_layer("new_regression_hidden").output

    plot_representation_from_output(
        stage3_model,
        latent_output,
        input_features,
        regression_targets,
        title="Stage 3 Best Tail-Refinement Representation Space"
    )


def run_three_stage_decoupled_training(seed=42):
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

    input_count = training_features.shape[1]

    print(f"Training samples: {len(training_features)}")
    print(f"Testing samples: {len(testing_features)}")

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
        10.0
    ]

    # -----------------------------
    # Stage 1 - Representation Learning
    # -----------------------------
    stage1_optimal_epoch_count = get_optimal_epoch_count_with_kfold(
        build_model_function=lambda: build_representation_learning_model(
            input_count
        ),
        training_features=training_features,
        training_regression_targets=training_regression_targets,
        training_tail_labels=training_tail_labels,
        tail_weight=1.0,
        fold_count=5,
        seed=seed
    )

    stage1_representation_model = build_representation_learning_model(
        input_count
    )
    compile_training_model(stage1_representation_model)

    stage1_representation_model = retrain_for_optimal_epoch_count(
        stage1_representation_model,
        training_features,
        training_regression_targets,
        training_tail_labels,
        stage1_optimal_epoch_count,
        tail_weight=1.0
    )

    stage1_training_overall_mae, stage1_training_tail_mae, stage1_training_aore = (
        evaluate_model_with_aore_components(
            stage1_representation_model,
            training_features,
            training_regression_targets,
            training_tail_labels
        )
    )

    print(f"Stage 1 optimal epoch count: {stage1_optimal_epoch_count}")
    print(f"Stage 1 training overall MAE: {stage1_training_overall_mae:.4f}")
    print(f"Stage 1 training rare MAE: {stage1_training_tail_mae:.4f}")
    print(f"Stage 1 training AORE: {stage1_training_aore:.4f}")

    plot_stage1_representation_space(
        stage1_representation_model,
        input_features,
        regression_targets
    )

    # -----------------------------
    # Stage 2 - Density-Aware Retraining
    # -----------------------------
    best_stage2_density_aware_model = None
    best_stage2_training_aore_score = np.inf
    best_stage2_tail_weight = None
    best_stage2_optimal_epoch_count = None

    for candidate_tail_weight in candidate_tail_weights:

        def build_stage2_density_aware_model():
            return create_retraining_model_with_new_heads(
                stage1_representation_model,
                trainable_feature_extractor=False
            )

        stage2_optimal_epoch_count = get_optimal_epoch_count_with_kfold(
            build_model_function=build_stage2_density_aware_model,
            training_features=training_features,
            training_regression_targets=training_regression_targets,
            training_tail_labels=training_tail_labels,
            tail_weight=candidate_tail_weight,
            fold_count=5,
            seed=seed
        )

        candidate_stage2_model = create_retraining_model_with_new_heads(
            stage1_representation_model,
            trainable_feature_extractor=False
        )

        compile_training_model(candidate_stage2_model)

        candidate_stage2_model = retrain_for_optimal_epoch_count(
            candidate_stage2_model,
            training_features,
            training_regression_targets,
            training_tail_labels,
            stage2_optimal_epoch_count,
            tail_weight=candidate_tail_weight
        )

        candidate_stage2_training_aore_score = evaluate_model_with_aore(
            candidate_stage2_model,
            training_features,
            training_regression_targets,
            training_tail_labels
        )

        print(
            "Stage 2 "
            f"rare weight={candidate_tail_weight}, "
            f"optimal epoch count={stage2_optimal_epoch_count}, "
            f"training AORE={candidate_stage2_training_aore_score:.4f}"
        )

        if candidate_stage2_training_aore_score < best_stage2_training_aore_score:
            best_stage2_training_aore_score = candidate_stage2_training_aore_score
            best_stage2_density_aware_model = candidate_stage2_model
            best_stage2_tail_weight = candidate_tail_weight
            best_stage2_optimal_epoch_count = stage2_optimal_epoch_count

    stage2_training_overall_mae, stage2_training_tail_mae, stage2_training_aore = (
        evaluate_model_with_aore_components(
            best_stage2_density_aware_model,
            training_features,
            training_regression_targets,
            training_tail_labels
        )
    )

    stage2_test_overall_mae, stage2_test_tail_mae, stage2_test_aore = (
        evaluate_model_with_aore_components(
            best_stage2_density_aware_model,
            testing_features,
            testing_regression_targets,
            testing_tail_labels
        )
    )

    print(f"Best Stage 2 rare weight: {best_stage2_tail_weight}")
    print(f"Best Stage 2 optimal epoch count: {best_stage2_optimal_epoch_count}")
    print(f"Best Stage 2 training overall MAE: {stage2_training_overall_mae:.4f}")
    print(f"Best Stage 2 training rare MAE: {stage2_training_tail_mae:.4f}")
    print(f"Best Stage 2 training AORE: {stage2_training_aore:.4f}")

    print("\nStage 2 Test Set Results")
    print(f"Stage 2 test overall MAE: {stage2_test_overall_mae:.4f}")
    print(f"Stage 2 test rare MAE: {stage2_test_tail_mae:.4f}")
    print(f"Stage 2 test AORE: {stage2_test_aore:.4f}")

    plot_stage2_representation_space(
        best_stage2_density_aware_model,
        input_features,
        regression_targets
    )

    # -----------------------------
    # Stage 3 - Tail Refinement
    # -----------------------------
    best_stage3_tail_refinement_model = None
    best_stage3_training_aore_score = np.inf
    best_stage3_tail_weight = None
    best_stage3_optimal_epoch_count = None

    for candidate_tail_weight in candidate_tail_weights:

        def build_stage3_tail_refinement_model():
            return create_retraining_model_with_new_heads(
                best_stage2_density_aware_model,
                trainable_feature_extractor=False
            )

        stage3_optimal_epoch_count = get_optimal_epoch_count_with_kfold(
            build_model_function=build_stage3_tail_refinement_model,
            training_features=training_features,
            training_regression_targets=training_regression_targets,
            training_tail_labels=training_tail_labels,
            tail_weight=candidate_tail_weight,
            fold_count=5,
            seed=seed
        )

        candidate_stage3_model = create_retraining_model_with_new_heads(
            best_stage2_density_aware_model,
            trainable_feature_extractor=False
        )

        compile_training_model(candidate_stage3_model)

        candidate_stage3_model = retrain_for_optimal_epoch_count(
            candidate_stage3_model,
            training_features,
            training_regression_targets,
            training_tail_labels,
            stage3_optimal_epoch_count,
            tail_weight=candidate_tail_weight
        )

        candidate_stage3_training_aore_score = evaluate_model_with_aore(
            candidate_stage3_model,
            training_features,
            training_regression_targets,
            training_tail_labels
        )

        print(
            "Stage 3 "
            f"rare weight={candidate_tail_weight}, "
            f"optimal epoch count={stage3_optimal_epoch_count}, "
            f"training AORE={candidate_stage3_training_aore_score:.4f}"
        )

        if candidate_stage3_training_aore_score < best_stage3_training_aore_score:
            best_stage3_training_aore_score = candidate_stage3_training_aore_score
            best_stage3_tail_refinement_model = candidate_stage3_model
            best_stage3_tail_weight = candidate_tail_weight
            best_stage3_optimal_epoch_count = stage3_optimal_epoch_count

    stage3_training_overall_mae, stage3_training_tail_mae, stage3_training_aore = (
        evaluate_model_with_aore_components(
            best_stage3_tail_refinement_model,
            training_features,
            training_regression_targets,
            training_tail_labels
        )
    )

    stage3_test_overall_mae, stage3_test_tail_mae, stage3_test_aore = (
        evaluate_model_with_aore_components(
            best_stage3_tail_refinement_model,
            testing_features,
            testing_regression_targets,
            testing_tail_labels
        )
    )

    print(f"Best Stage 3 rare weight: {best_stage3_tail_weight}")
    print(f"Best Stage 3 optimal epoch count: {best_stage3_optimal_epoch_count}")
    print(f"Best Stage 3 training overall MAE: {stage3_training_overall_mae:.4f}")
    print(f"Best Stage 3 training rare MAE: {stage3_training_tail_mae:.4f}")
    print(f"Best Stage 3 training AORE: {stage3_training_aore:.4f}")

    print("\nFinal Test Set Results")
    print(f"Best Stage 2 rare weight: {best_stage2_tail_weight}")
    print(f"Best Stage 3 rare weight: {best_stage3_tail_weight}")
    print(f"Stage 2 test overall MAE: {stage2_test_overall_mae:.4f}")
    print(f"Stage 2 test rare MAE: {stage2_test_tail_mae:.4f}")
    print(f"Stage 2 test AORE: {stage2_test_aore:.4f}")
    print(f"Stage 3 test overall MAE: {stage3_test_overall_mae:.4f}")
    print(f"Stage 3 test rare MAE: {stage3_test_tail_mae:.4f}")
    print(f"Stage 3 test AORE: {stage3_test_aore:.4f}")

    print("\nStage 3 Improvement Over Stage 2")
    print(
        f"Overall MAE improvement: "
        f"{stage2_test_overall_mae - stage3_test_overall_mae:.4f}"
    )
    print(
        f"Rare MAE improvement: "
        f"{stage2_test_tail_mae - stage3_test_tail_mae:.4f}"
    )
    print(
        f"AORE improvement: "
        f"{stage2_test_aore - stage3_test_aore:.4f}"
    )

    plot_stage3_representation_space(
        best_stage3_tail_refinement_model,
        input_features,
        regression_targets
    )

    return best_stage3_tail_refinement_model


if __name__ == "__main__":
    final_model = run_three_stage_decoupled_training(seed=42)
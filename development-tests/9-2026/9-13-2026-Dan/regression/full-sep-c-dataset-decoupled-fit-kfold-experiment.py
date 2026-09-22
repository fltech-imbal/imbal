import json
import os

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

import imbal
from aore_metric import AORE
from imbal.regression import reciprocal_importance


# ============================================================
# Controlled k-fold epoch/alpha experiment
#
# Phase 1:
#   For each k, run normal rRT_fit with k-fold validation using
#   ONE fixed seed. Record the final Stage 1/Stage 2 epoch
#   estimates and the selected alpha.
#
# Phase 2:
#   For each set of estimates, build a NEW model from the SAME
#   fixed seed and train on 100% of the training data with:
#       validation_split=None
#       epochs=(estimated_stage1, estimated_stage2)
#       sample_weight=weights for the selected alpha only
#
# This makes the controlled final models start from the same
# seeded model initialization while varying only the parameters
# estimated by each k-fold run.
# ============================================================

EXPERIMENT_SEED = 42
K_VALUES = [2, 3, 4, 5, 10]

target_column = "ln_peak_intensity"

max_epochs = 5000
batch_size = 32
threshold = np.log(10)
patience = 500

alpha_candidates = [0.2, 0.3, 0.4]

RESULTS_PATH = "saved_results/controlled_kfold_epoch_alpha_experiment.json"
MODEL_DIRECTORY = "saved_models/controlled_kfold_epoch_alpha"

os.makedirs("saved_results", exist_ok=True)
os.makedirs(MODEL_DIRECTORY, exist_ok=True)


# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv(
    "../../../../tutorials/data/SEP-C/sep_10mev_training.csv"
)
test_data = pd.read_csv(
    "../../../../tutorials/data/SEP-C/sep_10mev_testing.csv"
)

y_train = train_data[target_column].values.reshape(-1, 1).astype("float32")
y_test = test_data[target_column].values.reshape(-1, 1).astype("float32")

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test = test_data.drop(columns=[target_column]).values.astype(np.float32)


# ----------------------------
# Model
# ----------------------------
def build_model(input_shape: int) -> imbal.regression.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
    hidden2 = layers.Dense(12, activation="relu", name="hidden_layer2")(hidden1)
    hidden3 = layers.Dense(8, activation="relu", name="hidden_layer3")(hidden2)
    hidden4 = layers.Dense(6, activation="relu", name="hidden_layer4")(hidden3)
    flatten = layers.Flatten(name="representation_flatten")(hidden4)
    outputs = layers.Dense(1, name="output_layer")(flatten)

    return imbal.regression.Model(
        inputs=inputs,
        outputs=outputs,
        name="sep_model",
    )


def compile_model(model):
    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        weighted_metrics=[AORE(threshold=threshold), "mae"],
        generate_decoder_branch=True,
    )


def evaluate_regression_model(model, x_values, y_values):
    evaluation = model.evaluate(
        x_values,
        y_values,
        verbose=0,
        return_dict=True,
    )

    predictions = model.predict(x_values, verbose=0).reshape(-1)
    y_true = y_values.reshape(-1)

    absolute_errors = np.abs(y_true - predictions)
    overall_mae = float(np.mean(absolute_errors))

    common_mask = y_true < threshold
    rare_mask = y_true >= threshold

    common_mae = float(np.mean(absolute_errors[common_mask]))
    rare_mae = float(np.mean(absolute_errors[rare_mask]))
    aore = float((overall_mae + rare_mae) / 2.0)

    return {
        "test_loss": float(evaluation["loss"]),
        "overall_mae": overall_mae,
        "common_mae": common_mae,
        "rare_mae": rare_mae,
        "aore": aore,
    }


def copy_weights(model):
    return [np.array(weight, copy=True) for weight in model.get_weights()]


def weights_identical(weights_a, weights_b):
    return (
        len(weights_a) == len(weights_b)
        and all(np.array_equal(a, b) for a, b in zip(weights_a, weights_b))
    )


# ----------------------------
# Shared weighting data
# ----------------------------
labels_kde = y_train.reshape(-1).copy()
kde = imbal.regression.fit_kde(labels_kde)
densities = imbal.regression.get_sample_densities(labels_kde, kde)

weight_candidates = reciprocal_importance(
    densities,
    alpha=alpha_candidates,
)

candidate_evaluation_weights = np.ones(len(y_train))


# ============================================================
# PHASE 1: NORMAL K-FOLD ESTIMATION
# ============================================================
estimated_configs = []

print("\n" + "=" * 72)
print("PHASE 1: K-FOLD ESTIMATION")
print(f"Fixed seed for every k: {EXPERIMENT_SEED}")
print("=" * 72)

for k in K_VALUES:
    print("\n" + "=" * 72)
    print(f"Estimating with k={k}")
    print("=" * 72)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(EXPERIMENT_SEED)

    model = build_model(x_train.shape[1])
    compile_model(model)

    stage_one_history, stage_two_history = model.rRT_fit(
        x_train,
        y_train,
        validation_split=k,
        candidate_evaluation_sample_weight=candidate_evaluation_weights,
        sample_weight=weight_candidates,
        batch_size=batch_size,
        epochs=max_epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                min_delta=0.01,
            )
        ],
        seed=EXPERIMENT_SEED,
    )

    # With the current model.py k-fold path, these returned histories are
    # the final full-training fits, whose lengths equal the estimated
    # epoch counts chosen from the folds.
    stage_one_epochs = int(len(stage_one_history.history["loss"]))
    stage_two_epochs = int(len(stage_two_history.history["loss"]))

    best_alpha_index = int(model.best_weight_index)
    best_alpha = float(alpha_candidates[best_alpha_index])

    reconstruction_lambda = None if model._reconstruction_lambda is None else float(model._reconstruction_lambda)

    kfold_evaluation = evaluate_regression_model(model, x_test, y_test)

    config = {
        "k": int(k),
        "stage_one_epochs": stage_one_epochs,
        "stage_two_epochs": stage_two_epochs,
        "best_alpha_index": best_alpha_index,
        "best_alpha": best_alpha,
        "reconstruction_lambda": reconstruction_lambda,
        "original_kfold_final_test": kfold_evaluation,
    }
    estimated_configs.append(config)

    print("\nK-fold estimate:")
    print(f"  Stage 1 epochs: {stage_one_epochs}")
    print(f"  Stage 2 epochs: {stage_two_epochs}")
    print(f"  Best alpha:     {best_alpha}")
    print(f"  Reconstruction lambda: {reconstruction_lambda}")
    print(f"  Original k-fold final AORE: {kfold_evaluation['aore']:.4f}")


# ============================================================
# PHASE 2: FRESH, CONTROLLED FULL-DATA RETRAINING
# ============================================================
controlled_results = []
reference_initial_weights = None

print("\n" + "=" * 72)
print("PHASE 2: CONTROLLED FULL-DATA RETRAINING")
print("No validation; fresh model; same seed for every k.")
print("=" * 72)

for config in estimated_configs:
    k = config["k"]
    stage_one_epochs = config["stage_one_epochs"]
    stage_two_epochs = config["stage_two_epochs"]
    best_alpha = config["best_alpha"]
    reconstruction_lambda = config["reconstruction_lambda"]

    print("\n" + "=" * 72)
    print(
        f"Controlled retrain for k={k}: "
        f"S1={stage_one_epochs}, S2={stage_two_epochs}, alpha={best_alpha}, lambda={reconstruction_lambda}"
    )
    print("=" * 72)

    # Reset all Keras state and use the SAME seed before constructing
    # every controlled final model.
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(EXPERIMENT_SEED)

    controlled_model = build_model(x_train.shape[1])
    compile_model(controlled_model)

    current_initial_weights = copy_weights(controlled_model)

    if reference_initial_weights is None:
        reference_initial_weights = current_initial_weights
        same_initialization = True
        print("Stored reference initial weights from first controlled model.")
    else:
        same_initialization = weights_identical(
            current_initial_weights,
            reference_initial_weights,
        )
        print(
            "Initial weights identical to first controlled model: "
            f"{same_initialization}"
        )

    if not same_initialization:
        raise RuntimeError(
            f"Controlled model for k={k} did not initialize identically."
        )

    # Reuse the exact decoder reconstruction lambda learned during this k-fold run.
    # This prevents the fresh no-validation model from estimating a different lambda.
    if reconstruction_lambda is None:
        raise RuntimeError(f"No reconstruction lambda was captured for k={k}.")
    controlled_model._reconstruction_lambda = np.float32(reconstruction_lambda)
    print(f"Reused k-fold reconstruction lambda: {controlled_model._reconstruction_lambda}")

    # Generate ONLY the weights for the alpha selected by this k-fold run.
    # Passing one 1-D weight vector prevents another alpha candidate search.
    selected_weights = reciprocal_importance(
        densities,
        alpha=best_alpha,
    )

    # No validation and no callbacks/early stopping:
    # train for exactly the Stage 1 and Stage 2 epoch estimates.
    controlled_stage_one_history, controlled_stage_two_history = (
        controlled_model.rRT_fit(
            x_train,
            y_train,
            validation_split=None,
            sample_weight=selected_weights,
            batch_size=batch_size,
            epochs=(stage_one_epochs, stage_two_epochs),
            seed=EXPERIMENT_SEED,
        )
    )

    actual_stage_one_epochs = int(
        len(controlled_stage_one_history.history["loss"])
    )
    actual_stage_two_epochs = int(
        len(controlled_stage_two_history.history["loss"])
    )

    evaluation = evaluate_regression_model(
        controlled_model,
        x_test,
        y_test,
    )

    model_path = os.path.join(
        MODEL_DIRECTORY,
        f"controlled_k{k}_seed{EXPERIMENT_SEED}.keras",
    )
    controlled_model.save(model_path)

    result = {
        "k": int(k),
        "seed": int(EXPERIMENT_SEED),
        "estimated_stage_one_epochs": int(stage_one_epochs),
        "estimated_stage_two_epochs": int(stage_two_epochs),
        "actual_stage_one_epochs": actual_stage_one_epochs,
        "actual_stage_two_epochs": actual_stage_two_epochs,
        "best_alpha": float(best_alpha),
        "reconstruction_lambda": float(reconstruction_lambda),
        "same_initialization_as_reference": bool(same_initialization),
        "test_loss": evaluation["test_loss"],
        "overall_mae": evaluation["overall_mae"],
        "common_mae": evaluation["common_mae"],
        "rare_mae": evaluation["rare_mae"],
        "aore": evaluation["aore"],
        "model_path": model_path,
    }
    controlled_results.append(result)

    print("\nControlled result:")
    print(f"  Stage 1 epochs: {actual_stage_one_epochs}")
    print(f"  Stage 2 epochs: {actual_stage_two_epochs}")
    print(f"  Alpha:          {best_alpha}")
    print(f"  Recon lambda:   {reconstruction_lambda}")
    print(f"  Overall MAE:    {evaluation['overall_mae']:.4f}")
    print(f"  Common MAE:     {evaluation['common_mae']:.4f}")
    print(f"  Rare MAE:       {evaluation['rare_mae']:.4f}")
    print(f"  AORE:           {evaluation['aore']:.4f}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 100)
print("CONTROLLED EXPERIMENT SUMMARY")
print("=" * 100)
print(
    f"{'k':>3} | {'S1':>5} | {'S2':>5} | {'alpha':>5} | "
    f"{'orig AORE':>10} | {'ctrl AORE':>10} | {'ctrl rare':>10}"
)
print("-" * 100)

for config, result in zip(estimated_configs, controlled_results):
    print(
        f"{config['k']:>3} | "
        f"{config['stage_one_epochs']:>5} | "
        f"{config['stage_two_epochs']:>5} | "
        f"{config['best_alpha']:>5.1f} | "
        f"{config['original_kfold_final_test']['aore']:>10.4f} | "
        f"{result['aore']:>10.4f} | "
        f"{result['rare_mae']:>10.4f}"
    )

output = {
    "experiment_seed": int(EXPERIMENT_SEED),
    "k_values": [int(k) for k in K_VALUES],
    "alpha_candidates": [float(a) for a in alpha_candidates],
    "phase_1_kfold_estimates": estimated_configs,
    "phase_2_controlled_retraining": controlled_results,
}

with open(RESULTS_PATH, "w") as file:
    json.dump(output, file, indent=4)

print(f"\nResults saved to: {RESULTS_PATH}")
print(f"Controlled models saved to: {MODEL_DIRECTORY}")

import json
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

import imbal
from aore_metric import AORE
from imbal.regression import reciprocal_importance


# ============================================================
# Controlled rerun of the five already-estimated configurations.
#
# This is intentionally based on the original controlled experiment.
# It DOES NOT rerun k-fold estimation. It directly repeats Phase 2
# using the saved S1/S2 epoch counts, alpha values, and reconstruction
# lambdas, then uses imbal's own regression visualization functions.
# ============================================================

EXPERIMENT_SEED = 42
K_VALUES = [2, 3, 4, 5, 10]

target_column = "ln_peak_intensity"

# Keep the original experiment parameters unchanged.
max_epochs = 5000
batch_size = 32
threshold = np.log(10)
patience = 500

alpha_candidates = [0.2, 0.3, 0.4]

RUN_CONFIGS = [
    {"k": 2,  "stage_one_epochs": 260, "stage_two_epochs": 112,
     "best_alpha": 0.2, "reconstruction_lambda": 7.1967, "ctrl_aore": 1.7061},
    {"k": 3,  "stage_one_epochs": 59,  "stage_two_epochs": 713,
     "best_alpha": 0.4, "reconstruction_lambda": 5.7605, "ctrl_aore": 1.1523},
    {"k": 4,  "stage_one_epochs": 356, "stage_two_epochs": 336,
     "best_alpha": 0.2, "reconstruction_lambda": 8.7156, "ctrl_aore": 1.7057},
    {"k": 5,  "stage_one_epochs": 337, "stage_two_epochs": 459,
     "best_alpha": 0.3, "reconstruction_lambda": 8.6749, "ctrl_aore": 1.6299},
    {"k": 10, "stage_one_epochs": 225, "stage_two_epochs": 77,
     "best_alpha": 0.2, "reconstruction_lambda": 7.8554, "ctrl_aore": 1.7486},
]

RESULTS_PATH = "saved_results/controlled_rerun_plot_experiment.json"
MODEL_DIRECTORY = "saved_models/controlled_rerun_plots"
PLOT_DIRECTORY = "saved_plots/controlled_rerun_plots"

os.makedirs("saved_results", exist_ok=True)
os.makedirs(MODEL_DIRECTORY, exist_ok=True)
os.makedirs(PLOT_DIRECTORY, exist_ok=True)


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
# Model -- unchanged from original controlled experiment
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
        "predictions": predictions,
    }


def copy_weights(model):
    return [np.array(weight, copy=True) for weight in model.get_weights()]


def weights_identical(weights_a, weights_b):
    return (
        len(weights_a) == len(weights_b)
        and all(np.array_equal(a, b) for a, b in zip(weights_a, weights_b))
    )


# ----------------------------
# imbal plotting wrappers
#
# imbal itself creates the figures. We temporarily suppress plt.show()
# only so we can add the requested Ctrl AORE title before saving.
# ----------------------------
def save_imbal_true_vs_predictions(y_true, y_pred, config):
    output_path = os.path.join(
        PLOT_DIRECTORY,
        f"predicted_vs_actual_k{config['k']}_ctrl_aore_{config['ctrl_aore']:.4f}.png",
    )

    old_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        imbal.regression.plot_true_vs_predictions(
            y_true,
            y_pred,
        )
        fig = plt.gcf()
        fig.suptitle(
            f"k={config['k']} | Ctrl AORE = {config['ctrl_aore']:.4f}"
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    finally:
        plt.show = old_show

    print(f"Saved imbal predicted-vs-actual plot: {output_path}")
    return output_path


def save_imbal_tsne(model, x_values, y_values, config):
    output_path = os.path.join(
        PLOT_DIRECTORY,
        f"tsne_k{config['k']}_ctrl_aore_{config['ctrl_aore']:.4f}.png",
    )

    old_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        # representation_flatten is layer -2 in the original model.
        # Let imbal locate/extract that representation and perform t-SNE.
        imbal.regression.tsne_visualization(
            model,
            x_values,
            y_values.reshape(-1),
            representation_layer_index=-2,
            perplexity=50,
        )
        fig = plt.gcf()
        fig.suptitle(
            f"k={config['k']} | Ctrl AORE = {config['ctrl_aore']:.4f}"
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    finally:
        plt.show = old_show

    print(f"Saved imbal t-SNE plot: {output_path}")
    return output_path


# ----------------------------
# Shared weighting data -- same imbal path as original experiment
# ----------------------------
labels_kde = y_train.reshape(-1).copy()
kde = imbal.regression.fit_kde(labels_kde)
densities = imbal.regression.get_sample_densities(labels_kde, kde)


# ============================================================
# CONTROLLED FULL-DATA RERUNS
# ============================================================
controlled_results = []
reference_initial_weights = None

print("\n" + "=" * 90)
print("CONTROLLED FULL-DATA RERUNS FOR PAPER FIGURES")
print("No k-fold estimation; same seed and Phase-2 training setup as original.")
print(f"Fixed seed for every model: {EXPERIMENT_SEED}")
print(f"Batch size: {batch_size}")
print("=" * 90)

for config in RUN_CONFIGS:
    k = config["k"]
    stage_one_epochs = config["stage_one_epochs"]
    stage_two_epochs = config["stage_two_epochs"]
    best_alpha = config["best_alpha"]
    reconstruction_lambda = config["reconstruction_lambda"]

    print("\n" + "=" * 90)
    print(
        f"Controlled rerun for k={k}: "
        f"S1={stage_one_epochs}, S2={stage_two_epochs}, "
        f"alpha={best_alpha}, lambda={reconstruction_lambda}"
    )
    print("=" * 90)

    # Exactly the same reset/initialization procedure used by original Phase 2.
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

    # Reuse the exact reconstruction lambda obtained for this k.
    controlled_model._reconstruction_lambda = np.float32(
        reconstruction_lambda
    )
    print(
        "Reused reconstruction lambda: "
        f"{controlled_model._reconstruction_lambda}"
    )

    # Same imbal reciprocal-importance function used by original Phase 2.
    selected_weights = reciprocal_importance(
        densities,
        alpha=best_alpha,
    )

    # Same full-data rRT_fit call as original Phase 2:
    # no validation, batch size 32, exact S1/S2 epoch counts, seed 42.
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

    predictions = evaluation["predictions"].reshape(-1, 1)

    prediction_plot_path = save_imbal_true_vs_predictions(
        y_test,
        predictions,
        config,
    )

    tsne_plot_path = save_imbal_tsne(
        controlled_model,
        x_test,
        y_test,
        config,
    )

    result = {
        "k": int(k),
        "seed": int(EXPERIMENT_SEED),
        "stage_one_epochs": actual_stage_one_epochs,
        "stage_two_epochs": actual_stage_two_epochs,
        "best_alpha": float(best_alpha),
        "reconstruction_lambda": float(reconstruction_lambda),
        "historical_ctrl_aore": float(config["ctrl_aore"]),
        "rerun_aore": evaluation["aore"],
        "aore_difference": float(
            evaluation["aore"] - config["ctrl_aore"]
        ),
        "test_loss": evaluation["test_loss"],
        "overall_mae": evaluation["overall_mae"],
        "common_mae": evaluation["common_mae"],
        "rare_mae": evaluation["rare_mae"],
        "same_initialization_as_reference": bool(same_initialization),
        "model_path": model_path,
        "prediction_plot_path": prediction_plot_path,
        "tsne_plot_path": tsne_plot_path,
    }
    controlled_results.append(result)

    print("\nControlled rerun result:")
    print(f"  Stage 1 epochs:       {actual_stage_one_epochs}")
    print(f"  Stage 2 epochs:       {actual_stage_two_epochs}")
    print(f"  Alpha:                {best_alpha}")
    print(f"  Reconstruction lambda:{reconstruction_lambda}")
    print(f"  Historical ctrl AORE: {config['ctrl_aore']:.4f}")
    print(f"  Rerun AORE:           {evaluation['aore']:.4f}")
    print(
        f"  Difference:           "
        f"{evaluation['aore'] - config['ctrl_aore']:+.6f}"
    )


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 108)
print("CONTROLLED RERUN SUMMARY")
print("=" * 108)
print(
    f"{'k':>3} | {'S1':>5} | {'S2':>5} | {'alpha':>5} | "
    f"{'lambda':>8} | {'ctrl AORE':>10} | {'rerun AORE':>10}"
)
print("-" * 108)

for result in controlled_results:
    print(
        f"{result['k']:>3} | "
        f"{result['stage_one_epochs']:>5} | "
        f"{result['stage_two_epochs']:>5} | "
        f"{result['best_alpha']:>5.1f} | "
        f"{result['reconstruction_lambda']:>8.4f} | "
        f"{result['historical_ctrl_aore']:>10.4f} | "
        f"{result['rerun_aore']:>10.4f}"
    )

output = {
    "experiment_seed": int(EXPERIMENT_SEED),
    "k_values": [int(k) for k in K_VALUES],
    "batch_size": int(batch_size),
    "alpha_candidates": [float(a) for a in alpha_candidates],
    "controlled_retraining": controlled_results,
}

with open(RESULTS_PATH, "w") as file:
    json.dump(output, file, indent=4)

print(f"\nResults saved to: {RESULTS_PATH}")
print(f"Controlled models saved to: {MODEL_DIRECTORY}")
print(f"Plots saved to: {PLOT_DIRECTORY}")

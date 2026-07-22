import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import layers
import os
import json
from aore_metric import AORE

import imbal

seeds = [42, 43, 44, 45, 46]

target_column = "ln_peak_intensity"

max_epochs = 500
batch_size = 32
threshold = np.log(10)
PATIENCE = 100

# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv("../../../../tutorials/data/SEP-C/sep_10mev_training.csv")
test_data = pd.read_csv("../../../../tutorials/data/SEP-C/sep_10mev_testing.csv")

y_train_full = train_data[target_column].values.reshape(-1, 1).astype("float32")
y_test = test_data[target_column].values.reshape(-1, 1).astype("float32")

x_train_full = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test = test_data.drop(columns=[target_column]).values.astype(np.float32)

feature_names = train_data.drop(columns=[target_column]).columns.tolist()

# ----------------------------
# Model
# ----------------------------
def build_model(input_shape: int) -> imbal.regression.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
    hidden2 = layers.Dense(12, activation="relu", name="hidden_layer2")(hidden1)
    hidden3 = layers.Dense(8, activation="relu", name="hidden_layer3")(hidden2)
    hidden4 = layers.Dense(6, activation="relu", name="hidden_layer4")(hidden3)
    outputs = layers.Dense(1, name="output_layer")(hidden4)
    built_model = imbal.regression.Model(
        inputs=inputs,
        outputs=outputs,
        name="sep_model",
    )
    return built_model


MODEL_SAVE_PATH = "saved_models/balanced-fit-model-val.keras"
MEDIAN_PARAMS_SAVE_PATH = (
    "saved_models/median_params_balanced_fit_regression-val.json"
)
MULTI_SEED_RESULTS_SAVE_PATH = (
    "saved_results/multi_seed_results_balanced_fit_regression-val.json"
)

os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_results", exist_ok=True)

from imbal.regression import reciprocal_importance

alpha_candidates = [0.2, 0.5, 0.8, 0.9, 1.0, 1.1]

all_seed_results = []
trained_model_paths = {}

for seed in seeds:
    print("\n" + "=" * 60)
    print(f"Running seed {seed}")
    print("=" * 60)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    # Fresh copies are required because split() replaces the local arrays.
    x_train = x_train_full.copy()
    y_train = y_train_full.copy()

    # ----------------------------
    # Validation Set
    # ----------------------------
    labels_kde = y_train.reshape(-1).copy()
    kde = imbal.regression.fit_kde(labels_kde)
    densities = imbal.regression.get_sample_densities(labels_kde, kde)

    weight_candidates = reciprocal_importance(
        densities,
        alpha=alpha_candidates,
    )

    (x_train, y_train, sw_candidates), (x_val, y_val, sw_val) = (
        imbal.regression.split(
            x_train,
            y_train,
            sample_weights=weight_candidates,
            test_size=0.2,
            seed=seed,
        )
    )

    candidate_evaluation_weights = np.ones(len(y_val))

    # ----------------------------
    # Training
    # ----------------------------
    model = build_model(x_train.shape[1])

    model.compile(
        loss="mean_squared_error",
        weighted_metrics=[AORE(threshold=threshold), "mae"],
        optimizer="adam",
    )

    model.balanced_fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val.reshape(-1, 1), sw_val),
        candidate_evaluation_sample_weight=candidate_evaluation_weights,
        sample_weight=sw_candidates,
        batch_size=batch_size,
        epochs=max_epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
            )
        ],
    )

    best_alpha_index = int(model.best_weight_index)
    best_alpha = float(alpha_candidates[best_alpha_index])

    # ----------------------------
    # Evaluation
    # ----------------------------
    results = model.evaluate(x_test, y_test, verbose=0)
    loss = float(results[0])
    overall_mae = float(results[-1])

    predictions = model.predict(x_test, verbose=0).reshape(-1)
    y_true = y_test.reshape(-1)

    common_mask = y_true < threshold
    rare_mask = y_true >= threshold

    common_mae = float(
        np.mean(np.abs(y_true[common_mask] - predictions[common_mask]))
    )
    rare_mae = float(
        np.mean(np.abs(y_true[rare_mask] - predictions[rare_mask]))
    )
    aore = float((common_mae + rare_mae) / 2.0)

    seed_model_path = f"saved_models/tmp_balanced-fit-model-val-seed-{seed}.keras"
    model.save(seed_model_path)
    trained_model_paths[int(seed)] = seed_model_path

    seed_result = {
        "seed": int(seed),
        "test_loss": loss,
        "overall_mae": overall_mae,
        "common_mae": common_mae,
        "rare_mae": rare_mae,
        "aore": aore,
        "best_alpha_index": best_alpha_index,
        "best_alpha": best_alpha,
    }
    all_seed_results.append(seed_result)

    print(f"Test Loss: {loss:.4f}")
    print(f"Overall MAE: {overall_mae:.4f}")
    print(f"Common sample MAE (< ln(10)): {common_mae:.4f}")
    print(f"Rare sample MAE (>= ln(10)): {rare_mae:.4f}")
    print(f"AORE: {aore:.4f}")
    print(f"Best alpha index: {best_alpha_index}")
    print(f"Best alpha: {best_alpha}")

# ----------------------------
# Average results across all five seeds
# ----------------------------
average_overall_mae = float(
    np.mean([result["overall_mae"] for result in all_seed_results])
)
average_common_mae = float(
    np.mean([result["common_mae"] for result in all_seed_results])
)
average_rare_mae = float(
    np.mean([result["rare_mae"] for result in all_seed_results])
)
average_aore = float(
    np.mean([result["aore"] for result in all_seed_results])
)

# ----------------------------
# Select and save median-AORE model
# ----------------------------
sorted_results = sorted(
    all_seed_results,
    key=lambda result: result["aore"],
)
median_result = sorted_results[len(sorted_results) // 2]
median_seed = int(median_result["seed"])

median_model = keras.models.load_model(
    trained_model_paths[median_seed],
    custom_objects={
        "Model": imbal.regression.Model,
        "AORE": AORE,
    },
)

median_model.save(MODEL_SAVE_PATH)

median_params = {
    "median_seed": median_seed,
    "median_rank_by_aore": (len(sorted_results) // 2) + 1,
    "best_alpha_index": int(median_result["best_alpha_index"]),
    "best_alpha": float(median_result["best_alpha"]),
    "overall_mae": float(median_result["overall_mae"]),
    "common_mae": float(median_result["common_mae"]),
    "rare_mae": float(median_result["rare_mae"]),
    "aore": float(median_result["aore"]),
    "average_overall_mae_across_5_seeds": average_overall_mae,
    "average_common_mae_across_5_seeds": average_common_mae,
    "average_rare_mae_across_5_seeds": average_rare_mae,
    "average_aore_across_5_seeds": average_aore,
}

with open(MEDIAN_PARAMS_SAVE_PATH, "w") as file:
    json.dump(median_params, file, indent=4)

with open(MULTI_SEED_RESULTS_SAVE_PATH, "w") as file:
    json.dump(
        {
            "seeds": [int(seed) for seed in seeds],
            "all_seed_results": all_seed_results,
            "results_sorted_by_aore": sorted_results,
            "average_overall_mae": average_overall_mae,
            "average_common_mae": average_common_mae,
            "average_rare_mae": average_rare_mae,
            "average_aore": average_aore,
            "median_model": median_params,
        },
        file,
        indent=4,
    )

# Remove temporary per-seed model files after saving the median model.
for temporary_model_path in trained_model_paths.values():
    if os.path.exists(temporary_model_path):
        os.remove(temporary_model_path)

print("\n" + "=" * 60)
print("Multi-seed summary")
print("=" * 60)
print(f"Seeds run: {seeds}")
print(f"Average overall MAE: {average_overall_mae:.4f}")
print(f"Average common MAE: {average_common_mae:.4f}")
print(f"Average rare MAE: {average_rare_mae:.4f}")
print(f"Average AORE: {average_aore:.4f}")

print("\n" + "=" * 60)
print("Median model saved")
print("=" * 60)
print(f"Median seed: {median_seed}")
print(f"Median model AORE: {median_result['aore']:.4f}")
print(f"Best alpha: {median_result['best_alpha']}")
print(f"Model path: {MODEL_SAVE_PATH}")
print(f"Median parameters path: {MEDIAN_PARAMS_SAVE_PATH}")
print(f"Multi-seed results path: {MULTI_SEED_RESULTS_SAVE_PATH}")

# ----------------------------
# Evaluation and visualization of saved median model
# ----------------------------
predictions = median_model.predict(x_test, verbose=0)
y_true = y_test.reshape(-1)
y_pred = predictions.reshape(-1)

imbal.regression.plot_true_vs_predictions(
    y_test,
    predictions,
)

imbal.regression.tsne_visualization(
    median_model,
    x_test,
    y_test,
)

absolute_errors = np.abs(y_true - y_pred)

# Only consider rare samples.
rare_indices = np.where(y_true >= threshold)[0]
rare_errors = absolute_errors[rare_indices]

low_error_idx = rare_indices[np.argmin(rare_errors)]
large_error_idx = rare_indices[np.argmax(rare_errors)]

print("Low-error rare sample")
print("Index:", low_error_idx)
print("True:", y_true[low_error_idx])
print("Pred:", y_pred[low_error_idx])
print("Error:", absolute_errors[low_error_idx])

imbal.regression.lime_explain_tabular_sample(
    x_test[low_error_idx],
    median_model,
    x_train_full,
    figure_save_path="bal_fit_val_low_error.html",
    feature_names=feature_names,
)

print("Large-error rare sample")
print("Index:", large_error_idx)
print("True:", y_true[large_error_idx])
print("Pred:", y_pred[large_error_idx])
print("Error:", absolute_errors[large_error_idx])

imbal.regression.lime_explain_tabular_sample(
    x_test[large_error_idx],
    median_model,
    x_train_full,
    figure_save_path="bal_fit_val_large_error.html",
    feature_names=feature_names,
)

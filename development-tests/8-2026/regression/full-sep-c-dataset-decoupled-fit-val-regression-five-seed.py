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

max_epochs = 300
batch_size = 32
threshold = np.log(10)
PATIENCE = 100

# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv(
    "../../../tutorials/data/SEP-C/sep_10mev_training.csv"
)
test_data = pd.read_csv(
    "../../../tutorials/data/SEP-C/sep_10mev_testing.csv"
)

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



def evaluate_regression_model(model, x_values, y_values):
    """Evaluate the model and return loss, MAEs, AORE, and predictions."""
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


MODEL_SAVE_PATH = "saved_models/decoupled-fit-model-val.keras"
MEDIAN_PARAMS_SAVE_PATH = (
    "saved_models/median_params_decoupled_fit_regression-val.json"
)
MULTI_SEED_RESULTS_SAVE_PATH = (
    "saved_results/multi_seed_results_decoupled_fit_regression-val.json"
)

os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_results", exist_ok=True)

from imbal.regression import reciprocal_importance

alpha_candidates = [0.2, 0.5, 0.8, 0.9, 1.0, 1.1]

all_seed_results = []
temporary_model_paths = {}

for seed in seeds:
    print("\n" + "=" * 60)
    print(f"Running seed {seed}")
    print("=" * 60)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

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
        optimizer="adam",
        weighted_metrics=[AORE(threshold=threshold), "mae"],
    )

    model.rRT_fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val.reshape(-1, 1), sw_val),
        sample_weight=sw_candidates,
        candidate_evaluation_sample_weight=candidate_evaluation_weights,
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
    seed_evaluation = evaluate_regression_model(model, x_test, y_test)

    temporary_model_path = (
        f"saved_models/tmp_decoupled-fit-model-val-seed-{seed}.keras"
    )
    model.save(temporary_model_path)
    temporary_model_paths[int(seed)] = temporary_model_path

    seed_result = {
        "seed": int(seed),
        "test_loss": seed_evaluation["test_loss"],
        "overall_mae": seed_evaluation["overall_mae"],
        "common_mae": seed_evaluation["common_mae"],
        "rare_mae": seed_evaluation["rare_mae"],
        "aore": seed_evaluation["aore"],
        "best_alpha_index": best_alpha_index,
        "best_alpha": best_alpha,
    }
    all_seed_results.append(seed_result)

    print(f"Test Loss: {seed_evaluation['test_loss']:.4f}")
    print(f"Overall MAE: {seed_evaluation['overall_mae']:.4f}")
    print(f"Common sample MAE (< ln(10)): {seed_evaluation['common_mae']:.4f}")
    print(f"Rare sample MAE (>= ln(10)): {seed_evaluation['rare_mae']:.4f}")
    print(f"AORE: {seed_evaluation['aore']:.4f}")
    print(f"Best alpha index: {best_alpha_index}")
    print(f"Best alpha: {best_alpha}")

# ----------------------------
# Average results across all seeds
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
    temporary_model_paths[median_seed],
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

# Remove temporary per-seed models after saving the median model.
for temporary_model_path in temporary_model_paths.values():
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
    save_figure="temp.png",
)

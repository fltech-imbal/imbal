import json
import os
import shutil

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from aore_metric import AORE
import imbal


seeds = [42, 43, 44, 45, 46]

target_column = "ln_peak_intensity"

max_epochs = 300
batch_size = 32
threshold = np.log(10)

# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv(
    "../../../tutorials/data/SEP-C/sep_10mev_training.csv"
)
test_data = pd.read_csv(
    "../../../tutorials/data/SEP-C/sep_10mev_testing.csv"
)

y_train_full = (
    train_data[target_column]
    .values.reshape(-1, 1)
    .astype("float32")
)
y_test = (
    test_data[target_column]
    .values.reshape(-1, 1)
    .astype("float32")
)

x_train_full = (
    train_data.drop(columns=[target_column])
    .values.astype(np.float32)
)
x_test = (
    test_data.drop(columns=[target_column])
    .values.astype(np.float32)
)

# ----------------------------
# Model
# ----------------------------
def build_model(input_shape: int) -> imbal.regression.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(
        18,
        activation="relu",
        name="hidden_layer1",
    )(inputs)
    hidden2 = layers.Dense(
        12,
        activation="relu",
        name="hidden_layer2",
    )(hidden1)
    hidden3 = layers.Dense(
        8,
        activation="relu",
        name="hidden_layer3",
    )(hidden2)
    hidden4 = layers.Dense(
        6,
        activation="relu",
        name="hidden_layer4",
    )(hidden3)
    outputs = layers.Dense(
        1,
        name="output_layer",
    )(hidden4)

    return imbal.regression.Model(
        inputs=inputs,
        outputs=outputs,
        name="sep_model",
    )



def evaluate_regression_model(model, x_values, y_values):
    """Evaluate the model and return loss, MAEs, AORE, and predictions."""
    evaluation = model.evaluate(
        x_values,
        y_values,
        verbose=0,
        return_dict=True,
    )
    predictions = model.predict(
        x_values,
        verbose=0,
    ).reshape(-1)
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


MODEL_SAVE_PATH = "saved_models/regular-fit-model-val.keras"
MEDIAN_PARAMS_SAVE_PATH = (
    "saved_models/median_params_regular_fit_regression-val.json"
)
MULTI_SEED_RESULTS_SAVE_PATH = (
    "saved_results/multi_seed_results_regular_fit_regression-val.json"
)

# Use a method-specific temporary directory so other five-seed scripts
# cannot overwrite these seed models.
TEMP_MODEL_DIR = "saved_models/five_seed_temp_models"

os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_results", exist_ok=True)

# Start with an empty temporary directory.
if os.path.exists(TEMP_MODEL_DIR):
    shutil.rmtree(TEMP_MODEL_DIR)

os.makedirs(TEMP_MODEL_DIR, exist_ok=True)

all_seed_results = []
temporary_model_paths = {}

for seed in seeds:
    print("\n" + "=" * 60)
    print(f"Running seed {seed}")
    print("=" * 60)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    # Fresh copies are required because each seed receives its own
    # independently generated training/validation split.
    x_train = x_train_full.copy()
    y_train = y_train_full.copy()

    (x_train, y_train), (x_val, y_val) = imbal.regression.split(
        x_train,
        y_train,
        test_size=0.2,
        seed=seed,
    )

    model = build_model(x_train.shape[1])

    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        weighted_metrics=[
            AORE(threshold=threshold),
            "mae",
        ],
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(
            x_val,
            y_val.reshape(-1, 1),
        ),
        batch_size=batch_size,
        epochs=max_epochs,
    )

    # ----------------------------
    # Evaluation
    # ----------------------------
    seed_evaluation = evaluate_regression_model(
        model,
        x_test,
        y_test,
    )

    temporary_model_path = os.path.join(
        TEMP_MODEL_DIR,
        f"seed_{seed}.keras",
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
        "best_alpha_index": -1,
        "best_alpha": -1,
    }
    all_seed_results.append(seed_result)

    print(f"Test Loss: {seed_evaluation['test_loss']:.4f}")
    print(f"Overall MAE: {seed_evaluation['overall_mae']:.4f}")
    print(
        "Common sample MAE (< ln(10)): "
        f"{seed_evaluation['common_mae']:.4f}"
    )
    print(
        "Rare sample MAE (>= ln(10)): "
        f"{seed_evaluation['rare_mae']:.4f}"
    )
    print(f"AORE: {seed_evaluation['aore']:.4f}")

# ----------------------------
# Average results across all five seeds
# ----------------------------
average_overall_mae = float(
    np.mean(
        [
            result["overall_mae"]
            for result in all_seed_results
        ]
    )
)
average_common_mae = float(
    np.mean(
        [
            result["common_mae"]
            for result in all_seed_results
        ]
    )
)
average_rare_mae = float(
    np.mean(
        [
            result["rare_mae"]
            for result in all_seed_results
        ]
    )
)
average_aore = float(
    np.mean(
        [
            result["aore"]
            for result in all_seed_results
        ]
    )
)

# ----------------------------
# Select and save median-AORE model
# ----------------------------
sorted_results = sorted(
    all_seed_results,
    key=lambda result: result["aore"],
)

median_index = len(sorted_results) // 2
median_result = sorted_results[median_index]
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
    "median_rank_by_aore": median_index + 1,
    "best_alpha_index": -1,
    "best_alpha": -1,
    "overall_mae": float(
        median_result["overall_mae"]
    ),
    "common_mae": float(
        median_result["common_mae"]
    ),
    "rare_mae": float(
        median_result["rare_mae"]
    ),
    "aore": float(
        median_result["aore"]
    ),
    "average_overall_mae_across_5_seeds": (
        average_overall_mae
    ),
    "average_common_mae_across_5_seeds": (
        average_common_mae
    ),
    "average_rare_mae_across_5_seeds": (
        average_rare_mae
    ),
    "average_aore_across_5_seeds": average_aore,
}

with open(
    MEDIAN_PARAMS_SAVE_PATH,
    "w",
) as file:
    json.dump(
        median_params,
        file,
        indent=4,
    )

with open(
    MULTI_SEED_RESULTS_SAVE_PATH,
    "w",
) as file:
    json.dump(
        {
            "seeds": [
                int(seed)
                for seed in seeds
            ],
            "all_seed_results": all_seed_results,
            "results_sorted_by_aore": sorted_results,
            "average_overall_mae": (
                average_overall_mae
            ),
            "average_common_mae": (
                average_common_mae
            ),
            "average_rare_mae": (
                average_rare_mae
            ),
            "average_aore": average_aore,
            "median_model": median_params,
        },
        file,
        indent=4,
    )

# Remove all temporary seed models after the median model is saved.
shutil.rmtree(TEMP_MODEL_DIR)

print("\n" + "=" * 60)
print("Multi-seed summary")
print("=" * 60)
print(f"Seeds run: {seeds}")
print(
    "Average overall MAE: "
    f"{average_overall_mae:.4f}"
)
print(
    "Average common MAE: "
    f"{average_common_mae:.4f}"
)
print(
    "Average rare MAE: "
    f"{average_rare_mae:.4f}"
)
print(f"Average AORE: {average_aore:.4f}")

print("\n" + "=" * 60)
print("Median model saved")
print("=" * 60)
print(f"Median seed: {median_seed}")
print(
    "Median model AORE: "
    f"{median_result['aore']:.4f}"
)
print(f"Model path: {MODEL_SAVE_PATH}")
print(
    "Median parameters path: "
    f"{MEDIAN_PARAMS_SAVE_PATH}"
)
print(
    "Multi-seed results path: "
    f"{MULTI_SEED_RESULTS_SAVE_PATH}"
)

# ----------------------------
# Visualization of saved median model
# ----------------------------
median_predictions = median_model.predict(
    x_test,
    verbose=0,
)

imbal.regression.plot_true_vs_predictions(
    y_test,
    median_predictions,
    save_figure="temp.png",
)

import json
import os
import shutil

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

import imbal
from aore_metric import AORE


seeds = [42, 43, 44, 45, 46]

target_column = "ln_peak_intensity"

max_epochs = 500
batch_size = 32
patience = 100
threshold = np.log(10)

# Set this to True to skip training and load the previously saved median model.
LOAD_SAVED_MODEL = False

MODEL_SAVE_PATH = "saved_models/balanced-fit-model-val-ae.keras"
MEDIAN_PARAMS_SAVE_PATH = (
    "saved_models/median_params_balanced_fit_regression-val-ae.json"
)
MULTI_SEED_RESULTS_SAVE_PATH = (
    "saved_results/multi_seed_results_balanced_fit_regression-val-ae.json"
)
TEMP_MODEL_DIRECTORY = "saved_models/five_seed_temp_models_val_ae"

os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_results", exist_ok=True)
os.makedirs(TEMP_MODEL_DIRECTORY, exist_ok=True)

# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv(
    "../../../../tutorials/data/SEP-C/sep_10mev_training.csv"
)
test_data = pd.read_csv(
    "../../../../tutorials/data/SEP-C/sep_10mev_testing.csv"
)

y_train_full = train_data[target_column].values.reshape(-1, 1).astype("float32")
y_test = test_data[target_column].values.reshape(-1, 1).astype("float32")

x_train_full = train_data.drop(columns=[target_column]).values.astype(np.float32)
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
    flatten = layers.Flatten()(hidden4)
    outputs = layers.Dense(1, name="output_layer")(flatten)
    return imbal.regression.Model(
        inputs=inputs,
        outputs=outputs,
        name="sep_model",
    )


def evaluate_regression_model(model, x_values, y_values):
    """Evaluate the model and return loss, MAEs, AORE, and predictions."""
    evaluation = model.evaluate(x_values, y_values, verbose=0, return_dict=True)
    predictions = model.predict(x_values, verbose=0).reshape(-1)
    y_true = y_values.reshape(-1)

    overall_mae = float(np.mean(np.abs(y_true - predictions)))

    common_mask = y_true < threshold
    rare_mask = y_true >= threshold

    common_mae = float(
        np.mean(np.abs(y_true[common_mask] - predictions[common_mask]))
    )
    rare_mae = float(
        np.mean(np.abs(y_true[rare_mask] - predictions[rare_mask]))
    )

    # AORE is the equally weighted average of common- and rare-region MAE.
    aore = float((common_mae + rare_mae) / 2.0)

    return {
        "test_loss": float(evaluation["loss"]),
        "overall_mae": overall_mae,
        "common_mae": common_mae,
        "rare_mae": rare_mae,
        "aore": aore,
        "predictions": predictions,
    }


if LOAD_SAVED_MODEL and os.path.exists(MODEL_SAVE_PATH):
    print(f"Loading saved median regression model from {MODEL_SAVE_PATH}")
    model = keras.models.load_model(
        MODEL_SAVE_PATH,
        custom_objects={
            "Model": imbal.regression.Model,
            "AORE": AORE,
        },
    )

    final_evaluation = evaluate_regression_model(model, x_test, y_test)

    print(f"Test Loss: {final_evaluation['test_loss']:.4f}")
    print(f"Test Overall MAE: {final_evaluation['overall_mae']:.4f}")
    print(
        "Common sample MAE (< ln(10)): "
        f"{final_evaluation['common_mae']:.4f}"
    )
    print(
        "Rare sample MAE (>= ln(10)): "
        f"{final_evaluation['rare_mae']:.4f}"
    )
    print(f"Test AORE: {final_evaluation['aore']:.4f}")

    imbal.regression.plot_true_vs_predictions(
        y_test,
        final_evaluation["predictions"].reshape(-1, 1),
    )

else:
    # ----------------------------
    # Shared weighting data
    # ----------------------------
    labels_kde = y_train_full.reshape(-1).copy()
    kde = imbal.regression.fit_kde(labels_kde)
    densities = imbal.regression.get_sample_densities(labels_kde, kde)

    from imbal.regression import reciprocal_importance

    alpha_candidates = [0.2, 0.5, 0.8, 0.9, 1.0, 1.1]
    weight_candidates_full = reciprocal_importance(
        densities,
        alpha=alpha_candidates,
    )

    all_seed_results = []

    # ----------------------------
    # Five-seed training
    # ----------------------------
    for seed in seeds:
        print("\n" + "=" * 60)
        print(f"Running seed {seed}")
        print("=" * 60)

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)

        # Recreate the validation split for each seed without overwriting the
        # complete training arrays needed by later runs.
        (x_train, y_train, sw_candidates), (x_val, y_val, sw_val) = (
            imbal.regression.split(
                x_train_full,
                y_train_full,
                sample_weights=weight_candidates_full,
                test_size=0.2,
                seed=seed,
            )
        )
        candidate_evaluation_weights = np.ones(len(y_val))

        model = build_model(x_train_full.shape[1])
        model.compile(
            loss="mean_squared_error",
            weighted_metrics=[AORE(threshold=threshold), "mae"],
            optimizer="adam",
            generate_decoder_branch=True,
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
                    patience=patience,
                    restore_best_weights=True,
                )
            ],
        )

        best_alpha_index = int(model.best_weight_index)
        best_alpha = float(alpha_candidates[best_alpha_index])

        seed_evaluation = evaluate_regression_model(model, x_test, y_test)
        temporary_model_path = os.path.join(
            TEMP_MODEL_DIRECTORY,
            f"balanced-fit-model-val-ae-seed-{seed}.keras",
        )
        model.save(temporary_model_path)

        seed_result = {
            "seed": int(seed),
            "best_alpha_index": best_alpha_index,
            "best_alpha": best_alpha,
            "test_loss": seed_evaluation["test_loss"],
            "overall_mae": seed_evaluation["overall_mae"],
            "common_mae": seed_evaluation["common_mae"],
            "rare_mae": seed_evaluation["rare_mae"],
            "aore": seed_evaluation["aore"],
            "temporary_model_path": temporary_model_path,
        }
        all_seed_results.append(seed_result)

        print(f"Best alpha index: {best_alpha_index}")
        print(f"Best alpha: {best_alpha}")
        print(f"Test Loss: {seed_evaluation['test_loss']:.4f}")
        print(f"Test Overall MAE: {seed_evaluation['overall_mae']:.4f}")
        print(
            "Common sample MAE (< ln(10)): "
            f"{seed_evaluation['common_mae']:.4f}"
        )
        print(
            "Rare sample MAE (>= ln(10)): "
            f"{seed_evaluation['rare_mae']:.4f}"
        )
        print(f"Test AORE: {seed_evaluation['aore']:.4f}")

    # ----------------------------
    # Average results across all five seeds
    # ----------------------------
    average_test_loss = float(
        np.mean([result["test_loss"] for result in all_seed_results])
    )
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
    results_sorted_by_aore = sorted(
        all_seed_results,
        key=lambda result: result["aore"],
    )
    median_result = results_sorted_by_aore[len(results_sorted_by_aore) // 2]

    shutil.copy2(
        median_result["temporary_model_path"],
        MODEL_SAVE_PATH,
    )

    median_model_data = {
        "selection_metric": "aore",
        "selection_method": "median of five runs sorted by AORE",
        "median_seed": int(median_result["seed"]),
        "best_alpha_index": int(median_result["best_alpha_index"]),
        "best_alpha": float(median_result["best_alpha"]),
        "median_model_test_loss": float(median_result["test_loss"]),
        "median_model_overall_mae": float(median_result["overall_mae"]),
        "median_model_common_mae": float(median_result["common_mae"]),
        "median_model_rare_mae": float(median_result["rare_mae"]),
        "median_model_aore": float(median_result["aore"]),
        "average_test_loss_across_5_seeds": average_test_loss,
        "average_overall_mae_across_5_seeds": average_overall_mae,
        "average_common_mae_across_5_seeds": average_common_mae,
        "average_rare_mae_across_5_seeds": average_rare_mae,
        "average_aore_across_5_seeds": average_aore,
    }

    with open(MEDIAN_PARAMS_SAVE_PATH, "w") as file:
        json.dump(median_model_data, file, indent=4)

    # Remove local file paths from the portable results JSON.
    serializable_seed_results = []
    for result in all_seed_results:
        serializable_result = result.copy()
        serializable_result.pop("temporary_model_path")
        serializable_seed_results.append(serializable_result)

    with open(MULTI_SEED_RESULTS_SAVE_PATH, "w") as file:
        json.dump(
            {
                "seeds": [int(seed) for seed in seeds],
                "all_seed_results": serializable_seed_results,
                "average_test_loss_across_5_seeds": average_test_loss,
                "average_overall_mae_across_5_seeds": average_overall_mae,
                "average_common_mae_across_5_seeds": average_common_mae,
                "average_rare_mae_across_5_seeds": average_rare_mae,
                "average_aore_across_5_seeds": average_aore,
                "median_model": median_model_data,
            },
            file,
            indent=4,
        )

    print("\n" + "=" * 60)
    print("Five-seed summary")
    print("=" * 60)
    print(f"Seeds run: {seeds}")
    print(f"Average Test Loss: {average_test_loss:.4f}")
    print(f"Average Overall MAE: {average_overall_mae:.4f}")
    print(f"Average Common MAE: {average_common_mae:.4f}")
    print(f"Average Rare MAE: {average_rare_mae:.4f}")
    print(f"Average AORE: {average_aore:.4f}")

    print("\n" + "=" * 60)
    print("Median-AORE model saved")
    print("=" * 60)
    print(f"Median seed: {median_result['seed']}")
    print(f"Median model AORE: {median_result['aore']:.4f}")
    print(f"Median model path: {MODEL_SAVE_PATH}")
    print(f"Median params path: {MEDIAN_PARAMS_SAVE_PATH}")
    print(f"Multi-seed results path: {MULTI_SEED_RESULTS_SAVE_PATH}")

    # Load the saved median model for the final visualization.
    median_model = keras.models.load_model(
        MODEL_SAVE_PATH,
        custom_objects={
            "Model": imbal.regression.Model,
            "AORE": AORE,
        },
    )
    median_predictions = median_model.predict(x_test, verbose=0)

    imbal.regression.plot_true_vs_predictions(
        y_test,
        median_predictions,
    )

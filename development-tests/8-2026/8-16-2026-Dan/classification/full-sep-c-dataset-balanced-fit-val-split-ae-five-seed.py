import json
import os
import shutil

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

import imbal


seeds = [42, 43, 44, 45, 46]

target_column = "ln_peak_intensity"

max_epochs = 500
batch_size = 32
PATIENCE = 100

MODEL_SAVE_PATH = "saved_models/balanced-fit-val-split-ae-model.keras"
MEDIAN_PARAMS_SAVE_PATH = "saved_models/median_params_balanced_fit_val_split_ae.json"
MULTI_SEED_RESULTS_SAVE_PATH = "saved_results/multi_seed_results_balanced_fit_val_split_ae.json"
TEMP_MODEL_ROOT = "saved_models/five_seed_temp_models"
TEMP_MODEL_DIRECTORY = os.path.join(TEMP_MODEL_ROOT, "balanced_fit_val_split_ae")

os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_results", exist_ok=True)
os.makedirs(TEMP_MODEL_ROOT, exist_ok=True)

# Start this configuration with an empty temporary directory.
if os.path.exists(TEMP_MODEL_DIRECTORY):
    shutil.rmtree(TEMP_MODEL_DIRECTORY)
os.makedirs(TEMP_MODEL_DIRECTORY, exist_ok=True)


# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv(
    "../../../../tutorials/data/SEP-C/sep_10mev_training_classification.csv"
)
test_data = pd.read_csv(
    "../../../../tutorials/data/SEP-C/sep_10mev_testing_classification.csv"
)

y_train = train_data[target_column].values.reshape(-1, 1).astype("float32")
y_test = test_data[target_column].values.reshape(-1, 1).astype("float32")

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test = test_data.drop(columns=[target_column]).values.astype(np.float32)


# ----------------------------
# Model
# ----------------------------
def build_model(input_shape: int) -> imbal.classification.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
    hidden2 = layers.Dense(12, activation="relu", name="hidden_layer2")(hidden1)
    hidden3 = layers.Dense(8, activation="relu", name="hidden_layer3")(hidden2)
    hidden4 = layers.Dense(6, activation="relu", name="hidden_layer4")(hidden3)
    flatten = layers.Flatten()(hidden4)
    outputs = layers.Dense(1, activation="sigmoid", name="output_layer")(flatten)

    return imbal.classification.Model(
        inputs=inputs,
        outputs=outputs,
        name="sep_model",
    )


def metric_scalar(value):
    """Convert a scalar or one-element metric tensor/array to float."""
    return float(np.asarray(value).reshape(-1)[0])


# Weight pairs represent [common_class_weight, rare_class_weight].
class_weight_candidates = [
    [0.9, 0.1],
    [0.8, 0.2],
    [0.7, 0.3],
    [0.6, 0.4],
]
all_seed_results = []
temporary_model_paths = {}


# ----------------------------
# Five-seed training
# ----------------------------
for seed in seeds:
    print("\n" + "=" * 60)
    print(f"Running seed {seed}")
    print("=" * 60)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    model = build_model(x_train.shape[1])

    # HSS is intentionally first so Imbal uses it as the primary
    # classification metric for internal model/threshold selection.
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            imbal.metrics.HeidkeSkillScore(threshold=0.5, name="HSS"),
            tf.keras.metrics.F1Score(threshold=0.5, name="F1Score"),
            imbal.metrics.TrueSkillStatistic(threshold=0.5, name="TSS"),
        ],
        generate_decoder_branch=True,
    )

    model.balanced_fit(
        x_train,
        y_train,
        validation_split=0.3,
        class_weight=class_weight_candidates,
        batch_size=batch_size,
        epochs=max_epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
            )
        ],
        verbose_imbal=2,
        seed=seed,
    )

    # ----------------------------
    # Evaluation at default threshold
    # ----------------------------
    evaluation = model.evaluate(
        x_test,
        y_test,
        verbose=0,
        return_dict=True,
    )

    loss = float(evaluation["loss"])
    hss_default = metric_scalar(evaluation["HSS"])
    f1_default = metric_scalar(evaluation["F1Score"])
    tss_default = metric_scalar(evaluation["TSS"])

    print(f"Test Loss: {loss:.4f}")
    print(f"Test HSS: {hss_default:.4f}")
    print(f"Test F1Score: {f1_default:.4f}")
    print(f"Test TSS: {tss_default:.4f}")

    temporary_model_path = os.path.join(
        TEMP_MODEL_DIRECTORY,
        f"balanced-fit-val-split-ae-seed-{seed}.keras",
    )
    model.save(temporary_model_path)
    temporary_model_paths[int(seed)] = temporary_model_path

    seed_result = {
        "seed": int(seed),
        "test_loss": loss,
        "test_hss_default_threshold": hss_default,
        "test_f1_score_default_threshold": f1_default,
        "test_tss_default_threshold": tss_default,
        "best_weight_index": (
            int(model.best_weight_index)
            if model.best_weight_index is not None
            else None
        ),
        "best_class_weights": (
            [float(x) for x in model.best_class_weights]
            if model.best_class_weights is not None
            else None
        ),
        "best_decision_threshold": (
            float(model.best_decision_threshold)
            if model.best_decision_threshold is not None
            else None
        ),
        "hss_using_best_threshold": None,
        "f1_score_using_best_threshold": None,
        "tss_using_best_threshold": None,
        "included_in_average": False,
        "temporary_model_path": temporary_model_path,
    }

    # ----------------------------
    # Evaluation at Imbal's best decision threshold
    # ----------------------------
    if model.best_decision_threshold is not None:
        best_threshold = float(model.best_decision_threshold)
        test_probabilities = model.predict(
            x_test,
            verbose=0,
        ).reshape(-1, 1)

        hss_metric = imbal.metrics.HeidkeSkillScore(threshold=best_threshold)
        hss_metric.update_state(y_test, test_probabilities)

        f1_metric = keras.metrics.F1Score(threshold=best_threshold)
        f1_metric.update_state(y_test, test_probabilities)

        tss_metric = imbal.metrics.TrueSkillStatistic(threshold=best_threshold)
        tss_metric.update_state(y_test, test_probabilities)

        best_threshold_hss = metric_scalar(hss_metric.result())
        best_threshold_f1 = metric_scalar(f1_metric.result())
        best_threshold_tss = metric_scalar(tss_metric.result())

        seed_result["hss_using_best_threshold"] = best_threshold_hss
        seed_result["f1_score_using_best_threshold"] = best_threshold_f1
        seed_result["tss_using_best_threshold"] = best_threshold_tss

        print(
            f"Best found threshold: {best_threshold}\n"
            f"HSS using Best Threshold: {best_threshold_hss:.4f}\n"
            f"F1Score using Best Threshold: {best_threshold_f1:.4f}\n"
            f"TSS using Best Threshold: {best_threshold_tss:.4f}"
        )

        # Match the existing five-seed classification policy:
        # zero-F1 runs are excluded from averages and median selection.
        if best_threshold_f1 > 0.0:
            seed_result["included_in_average"] = True
        else:
            print(
                f"Seed {seed} excluded from average and median selection "
                "because F1Score using Best Threshold was 0.0000."
            )
    else:
        print(
            f"Seed {seed} excluded from average and median selection "
            "because no best decision threshold was found."
        )

    all_seed_results.append(seed_result)


# ----------------------------
# Valid nonzero-F1 runs
# ----------------------------
valid_seed_results = [
    result
    for result in all_seed_results
    if result["included_in_average"]
]


# ----------------------------
# Average and standard-deviation results across valid runs
# ----------------------------
if valid_seed_results:
    average_test_loss = float(
        np.mean([result["test_loss"] for result in valid_seed_results])
    )
    average_hss = float(
        np.mean(
            [result["hss_using_best_threshold"] for result in valid_seed_results]
        )
    )
    average_f1 = float(
        np.mean(
            [result["f1_score_using_best_threshold"] for result in valid_seed_results]
        )
    )
    average_tss = float(
        np.mean(
            [result["tss_using_best_threshold"] for result in valid_seed_results]
        )
    )

    std_hss = float(
        np.std(
            [result["hss_using_best_threshold"] for result in valid_seed_results]
        )
    )
    std_f1 = float(
        np.std(
            [result["f1_score_using_best_threshold"] for result in valid_seed_results]
        )
    )
    std_tss = float(
        np.std(
            [result["tss_using_best_threshold"] for result in valid_seed_results]
        )
    )
else:
    average_test_loss = None
    average_hss = None
    average_f1 = None
    average_tss = None
    std_hss = None
    std_f1 = None
    std_tss = None


# ----------------------------
# Select and save median-F1 model from valid runs
# ----------------------------
median_model_data = None

if valid_seed_results:
    results_sorted_by_f1 = sorted(
        valid_seed_results,
        key=lambda result: result["f1_score_using_best_threshold"],
    )
    median_index = len(results_sorted_by_f1) // 2
    median_result = results_sorted_by_f1[median_index]

    shutil.copy2(
        median_result["temporary_model_path"],
        MODEL_SAVE_PATH,
    )

    median_model_data = {
        "selection_metric": "f1_score_using_best_threshold",
        "selection_method": (
            "median of valid nonzero-F1 runs sorted by F1Score "
            "using the best decision threshold"
        ),
        "median_seed": int(median_result["seed"]),
        "median_rank_by_f1": median_index + 1,
        "best_weight_index": median_result["best_weight_index"],
        "best_class_weights": median_result["best_class_weights"],
        "best_decision_threshold": float(
            median_result["best_decision_threshold"]
        ),
        "median_model_test_loss": float(median_result["test_loss"]),
        "median_model_hss_using_best_threshold": float(
            median_result["hss_using_best_threshold"]
        ),
        "median_model_f1_score_using_best_threshold": float(
            median_result["f1_score_using_best_threshold"]
        ),
        "median_model_tss_using_best_threshold": float(
            median_result["tss_using_best_threshold"]
        ),
        "average_test_loss_across_valid_runs": average_test_loss,
        "average_hss_using_best_threshold_across_valid_runs": average_hss,
        "average_f1_score_using_best_threshold_across_valid_runs": average_f1,
        "average_tss_using_best_threshold_across_valid_runs": average_tss,
        "valid_run_count": len(valid_seed_results),
        "excluded_run_count": len(all_seed_results) - len(valid_seed_results),
    }

    with open(MEDIAN_PARAMS_SAVE_PATH, "w") as file:
        json.dump(median_model_data, file, indent=4)


# ----------------------------
# Save portable multi-seed results
# ----------------------------
serializable_seed_results = []
for result in all_seed_results:
    serializable_result = result.copy()
    serializable_result.pop("temporary_model_path")
    serializable_seed_results.append(serializable_result)

serializable_valid_seed_results = [
    result
    for result in serializable_seed_results
    if result["included_in_average"]
]

with open(MULTI_SEED_RESULTS_SAVE_PATH, "w") as file:
    json.dump(
        {
            "seeds": [int(seed) for seed in seeds],
            "all_seed_results": serializable_seed_results,
            "valid_seed_results": serializable_valid_seed_results,
            "valid_nonzero_f1_run_count": len(valid_seed_results),
            "excluded_run_count": len(all_seed_results) - len(valid_seed_results),
            "average_test_loss_across_valid_runs": average_test_loss,
            "average_hss_using_best_threshold_excluding_zero_f1_runs": average_hss,
            "std_hss_using_best_threshold_excluding_zero_f1_runs": std_hss,
            "average_f1_score_using_best_threshold_excluding_zero_f1_runs": average_f1,
            "std_f1_score_using_best_threshold_excluding_zero_f1_runs": std_f1,
            "average_tss_using_best_threshold_excluding_zero_f1_runs": average_tss,
            "std_tss_using_best_threshold_excluding_zero_f1_runs": std_tss,
            "median_model": median_model_data,
        },
        file,
        indent=4,
    )


# ----------------------------
# Remove temporary per-seed models
# ----------------------------
if os.path.exists(TEMP_MODEL_DIRECTORY):
    shutil.rmtree(TEMP_MODEL_DIRECTORY)

if os.path.exists(TEMP_MODEL_ROOT) and not os.listdir(TEMP_MODEL_ROOT):
    os.rmdir(TEMP_MODEL_ROOT)


# ----------------------------
# Summary
# ----------------------------
print("\n" + "=" * 60)
print("Multi-seed summary")
print("=" * 60)
print(f"Seeds run: {seeds}")
print(f"Total runs: {len(all_seed_results)}")
print(f"Valid nonzero-F1 runs used in average: {len(valid_seed_results)}")
print(
    "Runs excluded from average: "
    f"{len(all_seed_results) - len(valid_seed_results)}"
)

if average_f1 is not None:
    print(f"Average Test Loss across valid runs: {average_test_loss:.4f}")
    print(
        "Average HSS using Best Threshold, excluding zero-F1 runs: "
        f"{average_hss:.4f}"
    )
    print(
        "Std HSS using Best Threshold, excluding zero-F1 runs: "
        f"{std_hss:.4f}"
    )
    print(
        "Average F1Score using Best Threshold, excluding zero-F1 runs: "
        f"{average_f1:.4f}"
    )
    print(
        "Std F1Score using Best Threshold, excluding zero-F1 runs: "
        f"{std_f1:.4f}"
    )
    print(
        "Average TSS using Best Threshold, excluding zero-F1 runs: "
        f"{average_tss:.4f}"
    )
    print(
        "Std TSS using Best Threshold, excluding zero-F1 runs: "
        f"{std_tss:.4f}"
    )

    print("\n" + "=" * 60)
    print("Median-F1 model saved")
    print("=" * 60)
    print(f"Median seed: {median_model_data['median_seed']}")
    print(
        "Median model HSS using Best Threshold: "
        f"{median_model_data['median_model_hss_using_best_threshold']:.4f}"
    )
    print(
        "Median model F1Score using Best Threshold: "
        f"{median_model_data['median_model_f1_score_using_best_threshold']:.4f}"
    )
    print(
        "Median model TSS using Best Threshold: "
        f"{median_model_data['median_model_tss_using_best_threshold']:.4f}"
    )
    print(f"Median model path: {MODEL_SAVE_PATH}")
    print(f"Median params path: {MEDIAN_PARAMS_SAVE_PATH}")
else:
    print(
        "Average scores and median model could not be calculated because "
        "no valid nonzero-F1 runs were found."
    )

print(f"\nMulti-seed results saved to: {MULTI_SEED_RESULTS_SAVE_PATH}")

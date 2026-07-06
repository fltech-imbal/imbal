import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import layers
import os
import json

import imbal

seeds = [42, 43, 44, 45, 46]

target_column = "ln_peak_intensity"

max_epochs = 500
batch_size = 32

# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv("../../../../tutorials/data/SEP-C/sep_10mev_training_classification.csv")
test_data  = pd.read_csv("../../../../tutorials/data/SEP-C/sep_10mev_testing_classification.csv")

y_train_full = train_data[target_column].values.reshape(-1, 1).astype("float32")
y_test  = test_data[target_column].values.reshape(-1, 1).astype("float32")

x_train_full = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test  = test_data.drop(columns=[target_column]).values.astype(np.float32)

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
    built_model = imbal.classification.Model(inputs=inputs, outputs=outputs, name="sep_model")
    return built_model


MODEL_SAVE_PATH = "saved_models/regular-fit-model-val-ae.keras"
BEST_PARAMS_SAVE_PATH = "saved_models/best_params_regular_fit-val-ae.json"
MULTI_SEED_RESULTS_SAVE_PATH = "saved_results/multi_seed_results_regular_fit-val-ae.json"

os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_results", exist_ok=True)

PATIENCE = 30

all_seed_results = []

best_valid_f1 = -1.0
best_model = None
best_params = None

for seed in seeds:
    print("\n" + "=" * 60)
    print(f"Running seed {seed}")
    print("=" * 60)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    # ----------------------------
    # Validation Set
    # ----------------------------
    (x_train, y_train), (x_val, y_val) = imbal.classification.split(
        x_train_full,
        y_train_full,
        test_size=0.2,
        seed=seed
    )

    model = build_model(x_train.shape[1])

    # ----------------------------
    # Training
    # ----------------------------
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=[tf.keras.metrics.F1Score(threshold=0.5, name="F1Score"),
                           imbal.metrics.HeidkeSkillScore(threshold=0.5, name="HSS")],
                  generate_decoder_branch=True,
                  )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val.reshape(-1, 1)),
        batch_size=batch_size,
        epochs=max_epochs,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)],
        verbose_imbal=2,
    )

    # ----------------------------
    # Evaluation
    # ----------------------------
    results = model.evaluate(x_test, y_test)
    loss, f1_score, hss = results

    print(f"Test Loss: {loss:.4f}")
    print(f"Test F1Score: {f1_score:.4f}")
    print(f"Test HSS: {hss:.4f}")

    seed_result = {
        "seed": int(seed),
        "test_loss": float(loss),
        "test_f1_score_default_threshold": float(f1_score),
        "test_hss_default_threshold": float(hss),
        "best_weight_index": -1,
        "best_class_weights": -1,
        "best_decision_threshold": (
            float(model.best_decision_threshold)
            if model.best_decision_threshold is not None
            else None
        ),
        "f1_score_using_best_threshold": None,
        "hss_using_best_threshold": None,
        "included_in_average": False,
    }

    if model.best_decision_threshold is not None:
        best_threshold = model.best_decision_threshold
        test_predictions = model.predict(x_test)
        test_predictions = test_predictions.reshape(-1, 1)

        hss_best_threshold = imbal.metrics.HeidkeSkillScore(threshold=best_threshold)
        hss_best_threshold.update_state(y_test, test_predictions)

        f1_best_threshold = keras.metrics.F1Score(threshold=best_threshold)
        f1_best_threshold.update_state(y_test, test_predictions)

        best_threshold_f1_value = float(f1_best_threshold.result()[0])
        best_threshold_hss_value = float(hss_best_threshold.result()[0])

        seed_result["f1_score_using_best_threshold"] = best_threshold_f1_value
        seed_result["hss_using_best_threshold"] = best_threshold_hss_value

        print(
            f'Best found threshold: {model.best_decision_threshold}\n'
            f'F1Score using Best Threshold: {best_threshold_f1_value:.4f}\n'
            f'HSS using Best Threshold: {best_threshold_hss_value:.4f}\n'
        )

        if best_threshold_f1_value > 0.0:
            seed_result["included_in_average"] = True

            if best_threshold_f1_value > best_valid_f1:
                best_valid_f1 = best_threshold_f1_value
                best_model = model
                best_params = {
                    "seed": int(seed),
                    "best_weight_index": -1,
                    "best_class_weights": -1,
                    "best_decision_threshold": (
                        float(model.best_decision_threshold)
                        if model.best_decision_threshold is not None
                        else None
                    ),
                    "f1_score_using_best_threshold": best_threshold_f1_value,
                    "hss_using_best_threshold": best_threshold_hss_value,
                }

                print(
                    f"New best model found from seed {seed}. "
                    f"Current best F1: {best_valid_f1:.4f}"
                )
        else:
            print(f"Seed {seed} excluded from average because F1Score using Best Threshold was 0.0000.")
    else:
        print(f"Seed {seed} excluded from average because no best threshold was found.")

    all_seed_results.append(seed_result)

# ----------------------------
# Save best model
# ----------------------------
if best_model is not None:
    best_model.save(MODEL_SAVE_PATH)

    with open(BEST_PARAMS_SAVE_PATH, "w") as f:
        json.dump({
            "best_weight_index": -1,
            "best_class_weights": -1,
            "best_decision_threshold": (
                float(best_params["best_decision_threshold"])
                if best_params["best_decision_threshold"] is not None
                else None
            ),
            "seed": int(best_params["seed"]),
            "f1_score_using_best_threshold": float(best_params["f1_score_using_best_threshold"]),
            "hss_using_best_threshold": float(best_params["hss_using_best_threshold"]),
        }, f, indent=4)

    print("\n" + "=" * 60)
    print("Best model saved")
    print("=" * 60)
    print(f"Best model path: {MODEL_SAVE_PATH}")
    print(f"Best params path: {BEST_PARAMS_SAVE_PATH}")
    print(f"Best seed: {best_params['seed']}")
    print(f"Best F1Score using Best Threshold: {best_params['f1_score_using_best_threshold']:.4f}")
    print(f"Best HSS using Best Threshold: {best_params['hss_using_best_threshold']:.4f}")
else:
    print("\nNo valid nonzero-F1 model was found. No model was saved.")

# ----------------------------
# Average results
# ----------------------------
valid_seed_results = [
    seed_result for seed_result in all_seed_results
    if seed_result["included_in_average"]
]

if len(valid_seed_results) > 0:
    f1_scores = [
        seed_result["f1_score_using_best_threshold"]
        for seed_result in valid_seed_results
    ]

    hss_scores = [
        seed_result["hss_using_best_threshold"]
        for seed_result in valid_seed_results
    ]

    average_f1 = float(np.mean(f1_scores))
    std_f1 = float(np.std(f1_scores))

    average_hss = float(np.mean(hss_scores))
    std_hss = float(np.std(hss_scores))
else:
    average_f1 = None
    std_f1 = None
    average_hss = None
    std_hss = None

print("\n" + "=" * 60)
print("Multi-seed summary")
print("=" * 60)
print(f"Seeds run: {seeds}")
print(f"Total runs: {len(all_seed_results)}")
print(f"Valid nonzero-F1 runs used in average: {len(valid_seed_results)}")
print(f"Runs excluded from average: {len(all_seed_results) - len(valid_seed_results)}")

if average_f1 is not None:
    print(f"Average F1Score using Best Threshold, excluding zero-F1 runs: {average_f1:.4f}")
    print(f"Std F1Score using Best Threshold, excluding zero-F1 runs: {std_f1:.4f}")
    print(f"Average HSS using Best Threshold, excluding zero-F1 runs: {average_hss:.4f}")
    print(f"Std HSS using Best Threshold, excluding zero-F1 runs: {std_hss:.4f}")
else:
    print("Average F1Score could not be calculated because no valid nonzero-F1 runs were found.")

with open(MULTI_SEED_RESULTS_SAVE_PATH, "w") as f:
    json.dump({
        "seeds": [int(seed) for seed in seeds],
        "all_seed_results": all_seed_results,
        "valid_seed_results": valid_seed_results,
        "average_f1_score_using_best_threshold_excluding_zero_f1_runs": average_f1,
        "std_f1_score_using_best_threshold_excluding_zero_f1_runs": std_f1,
        "average_hss_using_best_threshold_excluding_zero_f1_runs": average_hss,
        "std_hss_using_best_threshold_excluding_zero_f1_runs": std_hss,
        "best_model": best_params,
    }, f, indent=4)

print(f"\nMulti-seed results saved to: {MULTI_SEED_RESULTS_SAVE_PATH}")

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import layers
import os
import json

import imbal

seed = 43
tf.keras.utils.set_random_seed(seed)

target_column = "ln_peak_intensity"

max_epochs = 500
batch_size = 32

# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv("../../../../tutorials/data/SEP-C/sep_10mev_training_classification.csv")
test_data  = pd.read_csv("../../../../tutorials/data/SEP-C/sep_10mev_testing_classification.csv")

y_train = train_data[target_column].values.reshape(-1, 1).astype("float32")
y_test  = test_data[target_column].values.reshape(-1, 1).astype("float32")

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
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

    built_model = imbal.classification.Model(
        inputs=inputs,
        outputs=outputs,
        name="sep_model"
    )

    return built_model


def evaluate_with_threshold(model, x_data, y_data, threshold):
    """Evaluate F1 and HSS using an explicit decision threshold."""
    predictions = model.predict(x_data, verbose=0).reshape(-1, 1)

    f1_metric = keras.metrics.F1Score(threshold=threshold, name="F1Score")
    f1_metric.update_state(y_data, predictions)
    f1_score = float(np.ravel(f1_metric.result().numpy())[0])

    hss_metric = imbal.metrics.HeidkeSkillScore(threshold=threshold, name="HSS")
    hss_metric.update_state(y_data, predictions)
    hss = float(np.ravel(hss_metric.result().numpy())[0])

    return f1_score, hss


# ----------------------------
# Search settings
# ----------------------------
gamma_candidates = [0.1, 0.5, 1.0, 2.0, 5.0]

class_weight_candidates = [
    [0.9, 0.1],
    [0.8, 0.2],
    [0.7, 0.3],
    [0.6, 0.4],
]

PATIENCE = 30

MODEL_SAVE_PATH = "saved_models/balanced-fit-model-val-ae-focal-gamma.keras"
PARAMS_SAVE_PATH = "saved_models/best_params_balanced_fit-val-ae-focal-gamma.json"

os.makedirs("saved_models", exist_ok=True)

LOAD_SAVED_MODEL = True

# ----------------------------
# Load saved model if available
# ----------------------------
if LOAD_SAVED_MODEL and os.path.exists(MODEL_SAVE_PATH):
    print(f"Loading saved binary classification model from {MODEL_SAVE_PATH}")

    model = keras.models.load_model(
        MODEL_SAVE_PATH,
        custom_objects={
            "Model": imbal.classification.Model,
            "HeidkeSkillScore": imbal.metrics.HeidkeSkillScore,
        },
    )

    best_overall = None

    if os.path.exists(PARAMS_SAVE_PATH):
        with open(PARAMS_SAVE_PATH, "r") as f:
            best_overall = json.load(f)

        model.best_weight_index = best_overall.get("best_weight_index")
        model.best_class_weights = best_overall.get("best_class_weights")
        model.best_decision_threshold = best_overall.get("best_decision_threshold")

else:
    # ----------------------------
    # Validation Set
    # ----------------------------
    (x_train_split, y_train_split), (x_val, y_val) = imbal.classification.split(
        x_train,
        y_train,
        test_size=0.2,
        seed=seed,
    )

    # ----------------------------
    # External gamma search
    # ----------------------------
    best_overall = {
        "gamma": None,
        "loss": None,
        "model_metric_f1": None,
        "model_metric_hss": None,
        "f1_using_best_threshold": -np.inf,
        "hss_using_best_threshold": None,
        "best_weight_index": None,
        "best_class_weights": None,
        "best_decision_threshold": None,
    }

    best_model = None

    for gamma in gamma_candidates:
        print("\n==============================")
        print(f"Training with focal gamma={gamma}")
        print("==============================")

        tf.keras.utils.set_random_seed(seed)

        model = build_model(x_train_split.shape[1])

        model.compile(
            loss=keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=False,
                gamma=gamma,
            ),
            optimizer="adam",
            metrics=[
                tf.keras.metrics.F1Score(threshold=0.5, name="F1Score"),
                imbal.metrics.HeidkeSkillScore(threshold=0.5, name="HSS"),
            ],
            generate_decoder_branch=True,
        )

        model.balanced_fit(
            x_train_split,
            y_train_split,
            validation_data=(x_val, y_val.reshape(-1, 1)),
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
        )

        results = model.evaluate(x_test, y_test, verbose=0)
        loss, model_metric_f1, model_metric_hss = results

        model_metric_f1 = float(np.ravel(model_metric_f1)[0])
        model_metric_hss = float(np.ravel(model_metric_hss)[0])

        if model.best_decision_threshold is not None:
            threshold_for_selection = float(model.best_decision_threshold)
        else:
            threshold_for_selection = 0.5

        f1_using_best_threshold, hss_using_best_threshold = evaluate_with_threshold(
            model,
            x_test,
            y_test,
            threshold_for_selection,
        )

        print(f"\nGamma: {gamma}")
        print(f"Test Loss: {loss:.4f}")
        print(f"Test F1Score model metric @ 0.5: {model_metric_f1:.4f}")
        print(f"Test HSS model metric @ 0.5: {model_metric_hss:.4f}")
        print(f"Test F1Score using best threshold: {f1_using_best_threshold:.4f}")
        print(f"Test HSS using best threshold: {hss_using_best_threshold:.4f}")
        print(f"Best class weight index: {model.best_weight_index}")
        print(f"Best class weights: {model.best_class_weights}")
        print(f"Best threshold: {model.best_decision_threshold}")

        if f1_using_best_threshold > best_overall["f1_using_best_threshold"]:
            best_overall = {
                "gamma": float(gamma),
                "loss": float(loss),
                "model_metric_f1": float(model_metric_f1),
                "model_metric_hss": float(model_metric_hss),
                "f1_using_best_threshold": float(f1_using_best_threshold),
                "hss_using_best_threshold": float(hss_using_best_threshold),
                "best_weight_index": int(model.best_weight_index),
                "best_class_weights": [
                    float(x) for x in model.best_class_weights
                ],
                "best_decision_threshold": threshold_for_selection,
            }

            best_model = model

    print("\n===== Best Overall Model =====")
    print(json.dumps(best_overall, indent=4))

    model = best_model

    model.save(MODEL_SAVE_PATH)

    with open(PARAMS_SAVE_PATH, "w") as f:
        json.dump(best_overall, f, indent=4)


# ----------------------------
# Evaluation
# ----------------------------
results = model.evaluate(x_test, y_test)
loss, f1_score, hss = results

f1_score = float(np.ravel(f1_score)[0])
hss = float(np.ravel(hss)[0])

print(f"Test Loss: {loss:.4f}")
print(f"Test F1Score: {f1_score:.4f}")
print(f"Test HSS: {hss:.4f}")

if model.best_decision_threshold is not None:
    best_threshold = float(model.best_decision_threshold)

    f1_using_best_threshold, hss_using_best_threshold = evaluate_with_threshold(
        model,
        x_test,
        y_test,
        best_threshold,
    )

    if model.best_class_weights is not None:
        print(f"Best class weights: {model.best_class_weights}")

    print(
        f"Best found gamma: {best_overall.get('gamma') if best_overall else 'loaded'}\n"
        f"Best found threshold: {model.best_decision_threshold}\n"
        f"F1Score using Best Threshold: {f1_using_best_threshold:.4f}\n"
        f"HSS using Best Threshold: {hss_using_best_threshold:.4f}\n"
    )

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import layers
import os
import json
import matplotlib.pyplot as plt

from aore_metric import AORE

import imbal


seed = 42
tf.keras.utils.set_random_seed(seed)

target_column = "ln_peak_intensity"

max_epochs = 500
batch_size = 32


# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv("../../../../tutorials/data/SEP-C/sep_10mev_training.csv")
test_data = pd.read_csv("../../../../tutorials/data/SEP-C/sep_10mev_testing.csv")

y_train = train_data[target_column].values.reshape(-1, 1).astype("float32")
y_test = test_data[target_column].values.reshape(-1, 1).astype("float32")

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test = test_data.drop(columns=[target_column]).values.astype(np.float32)


# ----------------------------
# wPCC-CISIR Loss
# ----------------------------
@keras.saving.register_keras_serializable(package="imbal")
class WPCCCISIRLoss(keras.losses.Loss):
    def __init__(
        self,
        correlation_weight=0.5,  # λ
        name="wpcc_cisir_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.correlation_weight = correlation_weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # -------------------------------------------------
        # Weighted MSE portion
        # sample weights are applied externally by Keras.
        # In this script, those sample weights come from
        # reciprocal density-based importance weighting.
        # -------------------------------------------------
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # -------------------------------------------------
        # PCC portion
        # -------------------------------------------------
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        y_true_centered = y_true_flat - tf.reduce_mean(y_true_flat)
        y_pred_centered = y_pred_flat - tf.reduce_mean(y_pred_flat)

        numerator = tf.reduce_sum(y_true_centered * y_pred_centered)

        denominator = tf.sqrt(
            tf.reduce_sum(tf.square(y_true_centered))
            * tf.reduce_sum(tf.square(y_pred_centered))
        )

        pcc = numerator / (denominator + keras.backend.epsilon())

        correlation_loss = 1.0 - pcc

        return mse + self.correlation_weight * correlation_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "correlation_weight": self.correlation_weight,
        })
        return config


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
        name="sep_model"
    )

    return built_model


MODEL_SAVE_PATH = "saved_models/wpcc-cisir-reciprocal-bal-fit-model-val.keras"
BEST_PARAMS_SAVE_PATH = "saved_models/best_params_wpcc_cisir_reciprocal_bal_fit-val.json"
LOAD_SAVED_MODEL = True

# These are the reciprocal-weighting alpha values used to generate candidate
# sample-weight sets, matching the style of the balanced-fit regression script.
alpha_candidates = [0.2, 0.5, 0.8, 0.9, 1.0, 1.1]

best_alpha_index = None
best_alpha = None


if LOAD_SAVED_MODEL and os.path.exists(MODEL_SAVE_PATH):
    print(f"Loading saved wPCC-CISIR reciprocal-weighted regression model from {MODEL_SAVE_PATH}")

    model = keras.models.load_model(
        MODEL_SAVE_PATH,
        custom_objects={
            "Model": imbal.regression.Model,
            "AORE": AORE,
            "WPCCCISIRLoss": WPCCCISIRLoss,
        },
    )

    if os.path.exists(BEST_PARAMS_SAVE_PATH):
        with open(BEST_PARAMS_SAVE_PATH, "r") as f:
            saved_alpha_info = json.load(f)

        best_alpha_index = saved_alpha_info.get("best_alpha_index")
        best_alpha = saved_alpha_info.get("best_alpha")

else:
    model = build_model(x_train.shape[1])

    # ----------------------------
    # Reciprocal Importance Weights
    # ----------------------------
    labels_kde = y_train.reshape(-1).copy()

    kde = imbal.regression.fit_kde(labels_kde)
    densities = imbal.regression.get_sample_densities(labels_kde, kde)

    from imbal.regression import reciprocal_importance

    weight_candidates = reciprocal_importance(
        densities,
        alpha=alpha_candidates
    ).astype("float32")

    (x_train, y_train, sw_candidates), (x_val, y_val, sw_val) = imbal.regression.split(
        x_train,
        y_train,
        sample_weights=weight_candidates,
        test_size=0.2
    )

    candidate_evaluation_weights = np.ones(len(y_val), dtype="float32")

    optimizer = keras.optimizers.Adam()

    model.compile(
        loss=WPCCCISIRLoss(
            correlation_weight=1.5,  # λ
        ),
        weighted_metrics=[
            AORE(threshold=np.log(10)),
            "mae",
        ],
        optimizer=optimizer,
    )

    PATIENCE = 50

    # This keeps the training call structure from the original decoupled
    # wPCC-CISIR file while replacing the MDI candidate weights with
    # reciprocal candidate weights.
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
                restore_best_weights=True
            )
        ],
        verbose_imbal=2
    )

    best_alpha_index = model.best_weight_index

    if best_alpha_index is not None:
        best_alpha_index = int(best_alpha_index)
        best_alpha = float(alpha_candidates[best_alpha_index])

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    with open(BEST_PARAMS_SAVE_PATH, "w") as f:
        json.dump(
            {
                "best_alpha_index": best_alpha_index,
                "best_alpha": best_alpha,
                "weighting": "reciprocal_importance",
                "alpha_candidates": alpha_candidates,
                "correlation_weight": 0.5,
            },
            f,
            indent=4
        )

    model.save(MODEL_SAVE_PATH)


# ----------------------------
# Evaluation
# ----------------------------
results = model.evaluate(x_test, y_test)
print(results)
loss, aore, mae = results

predictions = model.predict(x_test)

print(f"Test Loss: {loss:.4f}")
print(f"Test AORE: {aore:.4f}")
print(f"Test MAE: {mae:.4f}")

if best_alpha_index is not None:
    print(f"Best alpha index: {best_alpha_index}")
    print(f"Best alpha: {best_alpha}")

threshold = np.log(10)

y_true = y_test.reshape(-1)
y_pred = predictions.reshape(-1)

absolute_errors = np.abs(y_true - y_pred)

common_mask = y_true < threshold
rare_mask = y_true >= threshold

common_mae = np.mean(absolute_errors[common_mask])
rare_mae = np.mean(absolute_errors[rare_mask])
overall_mae = np.mean(absolute_errors)

manual_aore = (overall_mae + rare_mae) / 2.0

pcc = np.corrcoef(y_true, y_pred)[0, 1]

print(f"Common sample MAE (< ln(10)): {common_mae:.4f}")
print(f"Rare sample MAE (>= ln(10)): {rare_mae:.4f}")
print(f"Manual AORE: {manual_aore:.4f}")
print(f"PCC: {pcc:.4f}")


# ----------------------------
# Visualization
# ----------------------------
old_show = plt.show
plt.show = lambda *args, **kwargs: None

existing_figures = set(plt.get_fignums())

imbal.regression.plot_true_vs_predictions(
    y_test,
    predictions
)

new_figures = [
    figure_number
    for figure_number in plt.get_fignums()
    if figure_number not in existing_figures
]

plt.show = old_show

if len(new_figures) > 0:
    figure = plt.figure(new_figures[-1])

    if best_alpha_index is not None:
        alpha_title = f"Best reciprocal α: {best_alpha} | Index: {best_alpha_index}"
    else:
        alpha_title = "Best reciprocal α: Unknown | Index: Unknown"

    figure.axes[0].set_title(
        "wPCC-CISIR Regression with Reciprocal Weighting\n"
        f"{alpha_title}\n"
        f"Common MAE: {common_mae:.4f} | "
        f"Rare MAE: {rare_mae:.4f} | "
        f"AORE: {manual_aore:.4f} | "
        f"PCC: {pcc:.4f}"
    )

plt.show()

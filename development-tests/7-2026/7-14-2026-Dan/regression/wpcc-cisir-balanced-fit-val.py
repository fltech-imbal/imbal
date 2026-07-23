import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import layers
import os
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
train_data = pd.read_csv("sep_10mev_training_pseudo_labeled.csv")
test_data  = pd.read_csv("sep_10mev_testing_pseudo_labeled.csv")

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
        correlation_weight=0.5,  # λ from the paper
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
        # (sample weights are applied externally by Keras)
        # -------------------------------------------------
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # -------------------------------------------------
        # wPCC portion
        #
        # NOTE:
        # This implementation currently uses equal weights
        # (r_ci = 1 for all samples).
        #
        # To fully reproduce the SEP-C configuration from
        # the paper, αc = 1.7 would need to be used to
        # generate per-sample correlation weights and those
        # weights would need to be passed into this loss.
        # Keras losses only receive y_true/y_pred, so that
        # requires additional refactoring.
        # -------------------------------------------------
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        y_true_centered = y_true_flat - tf.reduce_mean(y_true_flat)
        y_pred_centered = y_pred_flat - tf.reduce_mean(y_pred_flat)

        numerator = tf.reduce_sum(
            y_true_centered * y_pred_centered
        )

        denominator = tf.sqrt(
            tf.reduce_sum(tf.square(y_true_centered))
            * tf.reduce_sum(tf.square(y_pred_centered))
        )

        pcc = numerator / (
            denominator + keras.backend.epsilon()
        )

        correlation_loss = 1.0 - pcc

        return (
            mse
            + self.correlation_weight * correlation_loss
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "correlation_weight": self.correlation_weight,
        })
        return config


# ----------------------------
# MDI Importance
# ----------------------------
def mdi_importance_from_densities(densities, alpha=1.0, epsilon=1e-6):
    densities = np.asarray(densities).reshape(-1)

    normalized_densities = densities / (np.max(densities) + 1e-3)
    normalized_densities = np.clip(normalized_densities, epsilon, 1.0 - epsilon)

    raw_weights = np.power(
        1.0 - np.power(normalized_densities, alpha),
        1.0 / alpha
    )

    mdi_weights = raw_weights / np.mean(raw_weights)

    return mdi_weights.astype("float32")


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


MODEL_SAVE_PATH = "saved_models/wpcc-cisir-fit-model-val.keras"
ALPHA_SAVE_PATH = MODEL_SAVE_PATH.replace(".keras", "_best_alpha.npy")
LOAD_SAVED_MODEL = True

mdi_alphas = [
    0.01,
    0.05,
    0.1,
    0.2,
    0.5,
    0.8,
    1.0,
    1.5,
    1.7,
    2.0,
    2.4,
    3.0,
    5.0,
]

best_alpha_index = None
best_alpha = None


if LOAD_SAVED_MODEL and os.path.exists(MODEL_SAVE_PATH):
    print(f"Loading saved wPCC-CISIR regression model from {MODEL_SAVE_PATH}")

    model = keras.models.load_model(
        MODEL_SAVE_PATH,
        custom_objects={
            "Model": imbal.regression.Model,
            "AORE": AORE,
            "WPCCCISIRLoss": WPCCCISIRLoss,
        },
    )

    if os.path.exists(ALPHA_SAVE_PATH):
        saved_alpha_info = np.load(ALPHA_SAVE_PATH, allow_pickle=True).item()
        best_alpha_index = saved_alpha_info["best_alpha_index"]
        best_alpha = saved_alpha_info["best_alpha"]

else:
    model = build_model(x_train.shape[1])

    labels_kde = y_train.reshape(-1).copy()

    kde = imbal.regression.fit_kde(labels_kde)
    densities = imbal.regression.get_sample_densities(labels_kde, kde)

    weight_candidates = np.array([
        mdi_importance_from_densities(
            densities,
            alpha=alpha
        )
        for alpha in mdi_alphas
    ]).astype("float32")

    (x_train, y_train, sw_candidates), (x_val, y_val, sw_val) = imbal.regression.split(
        x_train,
        y_train,
        sample_weights=weight_candidates,
        test_size=0.2
    )

    candidate_evaluation_weights = np.ones(len(y_val), dtype="float32")

    model.compile(
        loss=WPCCCISIRLoss(
            correlation_weight=0.3,  # λ
        ),
        weighted_metrics=[
            AORE(threshold=np.log(10)),
            "mae",
        ],
        optimizer="adam",
    )

    PATIENCE = 50

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
        best_alpha = mdi_alphas[best_alpha_index]

    np.save(
        ALPHA_SAVE_PATH,
        {
            "best_alpha_index": best_alpha_index,
            "best_alpha": best_alpha,
        }
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
        alpha_title = f"Best α: {best_alpha} | Index: {best_alpha_index}"
    else:
        alpha_title = "Best α: Unknown | Index: Unknown"

    figure.axes[0].set_title(
        "wPCC-CISIR Regression\n"
        f"{alpha_title}\n"
        f"Common MAE: {common_mae:.4f} | "
        f"Rare MAE: {rare_mae:.4f} | "
        f"AORE: {manual_aore:.4f} | "
        f"PCC: {pcc:.4f}"
    )

plt.show()
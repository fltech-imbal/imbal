
"""
One-fold training run that uses Keras EarlyStopping to identify the "ideal" epoch.

What you get:
- A single stratified fold split (train/val) from the provided training CSV.
- EarlyStopping on val_loss with (patience, min_delta).
- A small tracker callback to report the BEST epoch (1-based) and best val_loss,
  independent of when training actually stops.
- Optionally restores best weights (restore_best_weights=True).

Run:
  python one_fold_early_stop_epoch.py
"""

import time
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import layers

import imbal
from sklearn.model_selection import StratifiedKFold


# ----------------------------
# Config
# ----------------------------
seed = 42
tf.keras.utils.set_random_seed(
    seed
)

target_column = "ln_peak_intensity"
threshold = np.log(10.0)

num_folds_for_split = 5
max_epochs = 1000
batch_size = 512
seed = 42

# EarlyStopping rule:
early_stop_patience = 30
local_minima_patience = 30
early_stop_min_delta = 0.001  # e.g. 1e-4 for stricter improvements
local_minima_min_delta = 0.001  # e.g. 1e-4 for stricter improvements
monitor_metric = "val_loss"


# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_training.csv")
test_data = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_testing.csv")

y_train = train_data[target_column].values.astype(np.float32).reshape(-1, 1)
y_test  = test_data[target_column].values.astype(np.float32).reshape(-1, 1)

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test = test_data.drop(columns=[target_column]).values.astype(np.float32)

labels_kde = y_train.reshape(-1).copy()
kde = imbal.regression.fit_kde(labels_kde)
densities = imbal.regression.get_sample_densities(labels_kde, kde)
sample_weights = imbal.regression.generate_sample_weights(densities)


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
    model = imbal.regression.Model(inputs=inputs, outputs=outputs, name="one_hidden_layer_6_units")
    return model


# ----------------------------
# Callback: track BEST epoch for a monitored metric
# ----------------------------
class BestEpochTracker(keras.callbacks.Callback):
    """
    Tracks best value and best epoch (0-based internally, reported as 1-based).
    """
    def __init__(self, monitor="val_loss", mode="min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_epoch = None
        self.best_value = None

        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

    def on_train_begin(self, logs=None):
        self.best_epoch = None
        self.best_value = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.monitor not in logs:
            return

        current = float(logs[self.monitor])

        if self.best_value is None:
            self.best_value = current
            self.best_epoch = epoch
            return

        improved = (current < self.best_value) if self.mode == "min" else (current > self.best_value)
        if improved:
            self.best_value = current
            self.best_epoch = epoch


class LocalMinimumStopping(tf.keras.callbacks.Callback):
    """
    Stops training when val_loss rises by `delta` for `patience` consecutive epochs
    after a minimum has been observed.
    """
    def __init__(self, monitor="val_loss", delta=0.0, patience=5, restore_best_weights=True):
        super().__init__()
        self.monitor = monitor
        self.delta = delta
        self.patience = patience
        self.restore_best_weights = bool(restore_best_weights)

        self.best = None
        self.num_up = 0
        self.best_epoch = None
        self.stopped_epoch = None
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.best = None
        self.best_epoch = None
        self.num_up = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        # First epoch
        if self.best is None:
            self.best = current
            self.best_epoch = epoch + 1
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            return

        # New minimum
        if current < self.best - self.delta:
            self.best = current
            self.best_epoch = epoch + 1
            self.num_up = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            # Loss increased or flat
            self.num_up += 1

        if self.num_up >= self.patience:
            print(
                f"\nLocal minimum of {self.best:.4f} detected at epoch {self.best_epoch}, "
                f"stopping at epoch {epoch + 1} with loss of {current:.4f}"
            )
            self.model.stop_training = True
            self.stopped_epoch = epoch + 1
            if self.restore_best_weights and self.best_weights is not None:
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        # If training ended normally (no stop), still restore best if requested.
        if self.restore_best_weights and self.best_weights is not None and not self.model.stop_training:
            self.model.set_weights(self.best_weights)

# ----------------------------
# Train with EarlyStopping on val_loss
# ----------------------------
model = build_model(x_train.shape[1])

early_stop = keras.callbacks.EarlyStopping(
    monitor=monitor_metric,
    min_delta=early_stop_min_delta,
    patience=early_stop_patience,
    mode="min",
    restore_best_weights=True,
    verbose=1,
)

local_minima = LocalMinimumStopping(
    monitor="val_loss",
    delta=local_minima_min_delta,
    patience=local_minima_patience,
)

start_cpu = time.process_time()

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
              )

history = model.balanced_fit(
    x=x_train,
    y=y_train,
    sample_weight=sample_weights,
    epochs=max_epochs,
    batch_size=batch_size,
    #generate_decoder_branch=True,
    #representation_layer_index=-2,
    do_cross_validation=True,
    num_folds=num_folds_for_split,
    callbacks=[early_stop],
)

end_cpu = time.process_time()
print(f"\nCPU time spent (one-fold train): {end_cpu - start_cpu:.4f} seconds")

results = model.evaluate(x_test, y_test, verbose=0)

loss, f1_val, auroc_val = results
print("\n=== Model test results ===")
print(f"loss: {loss:.4f}")
print(f"f1_score: {f1_val:.4f}")
print(f"auroc: {auroc_val:.4f}")

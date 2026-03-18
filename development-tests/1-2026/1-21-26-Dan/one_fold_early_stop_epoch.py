
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
test_data  = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_testing.csv")

y_train = (train_data[target_column].values >= threshold).astype(int).reshape(-1, 1).astype("float32")
y_test  = (test_data[target_column].values  >= threshold).astype(int).reshape(-1, 1).astype("float32")

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test  = test_data.drop(columns=[target_column]).values.astype(np.float32)


# ----------------------------
# Model
# ----------------------------
def build_model(input_shape: int) -> keras.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
    hidden2 = layers.Dense(12, activation="relu", name="hidden_layer2")(hidden1)
    hidden3 = layers.Dense(8,  activation="relu", name="hidden_layer3")(hidden2)
    hidden4 = layers.Dense(6,  activation="relu", name="hidden_layer4")(hidden3)
    outputs = layers.Dense(1, activation="sigmoid", name="output_layer")(hidden4)
    return keras.Model(inputs=inputs, outputs=outputs, name="sep_model")


def make_compile_params():
    f1 = tf.keras.metrics.F1Score(threshold=0.5)
    return imbal.classification.wrap_model_compile_parameters(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=[f1],
    )


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
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            return

        # New minimum
        if current < self.best - self.delta:
            self.best = current
            self.best_epoch = epoch
            self.num_up = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            # Loss increased or flat
            self.num_up += 1

        if self.num_up >= self.patience:
            print(
                f"\nLocal minimum of {self.best:.4f} detected at epoch {self.best_epoch + 1}, "
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
# One-fold split (stratified)
# ----------------------------
y_train_1d = y_train.reshape(-1).astype(int)

kfold = StratifiedKFold(n_splits=num_folds_for_split, shuffle=True, random_state=seed)
tr_idx, va_idx = next(kfold.split(x_train, y_train_1d))

X_tr, X_va = x_train[tr_idx], x_train[va_idx]
y_tr, y_va = y_train[tr_idx], y_train[va_idx]


# ----------------------------
# Train with EarlyStopping on val_loss
# ----------------------------
model = build_model(x_train.shape[1])

sample_weights = imbal.classification.generate_sample_weights(y_tr)

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

tracker = BestEpochTracker(monitor=monitor_metric, mode="min")

print("------------------------------------------------------------------------")
print("One-fold training run (EarlyStopping on val_loss)")
print(f"Train size: {X_tr.shape[0]}  |  Val size: {X_va.shape[0]}")
print(f"max_epochs={max_epochs}, batch_size={batch_size}")
print(f"EarlyStopping: patience={early_stop_patience}, min_delta={early_stop_min_delta}, restore_best_weights=True")
print(f"Local Minima: patience={local_minima_patience}, min_delta={local_minima_min_delta}, restore_best_weights=True")
print("------------------------------------------------------------------------")

start_cpu = time.process_time()

history = imbal.classification.balanced_fit(
    model,
    X_tr,
    y_tr,
    sample_weights=sample_weights,
    compile_parameters=make_compile_params(),
    epochs=max_epochs,
    batch_size=batch_size,
    validation_data=(X_va, y_va),
    # callbacks=[early_stop, tracker],
    callbacks=[local_minima, tracker],
)
print(history)

end_cpu = time.process_time()
print(f"\nCPU time spent (one-fold train): {end_cpu - start_cpu:.4f} seconds")

hist = history.history

best_epoch_1based = (tracker.best_epoch + 1) if tracker.best_epoch is not None else None
best_val = tracker.best_value

print("\n------------------------------------------------------------------------")
print(f"BEST epoch by {monitor_metric}: {best_epoch_1based} (1-based)")
print(f"BEST {monitor_metric}: {best_val:.6f}" if best_val is not None else f"No {monitor_metric} found in logs.")
print(f"EarlyStopping stopped_epoch (1-based): {early_stop.stopped_epoch if early_stop.stopped_epoch else 'not stopped early'}")
print(f"LocalMinima stopped_epoch (1-based): {local_minima.best_epoch if local_minima.stopped_epoch else 'not stopped early'}")
print("------------------------------------------------------------------------")

# Evaluate on val and holdout test set using the restored best weights
val_scores = model.evaluate(X_va, y_va, verbose=0)
val_metric_dict = dict(zip(model.metrics_names, val_scores))

test_scores = model.evaluate(x_test, y_test, verbose=0)
test_metric_dict = dict(zip(model.metrics_names, test_scores))

print("\n=== Validation results (best weights restored) ===")
for k, v in val_metric_dict.items():
    print(f"{k}: {float(v):.4f}")

print("\n=== Holdout test results (best weights restored) ===")
for k, v in test_metric_dict.items():
    print(f"{k}: {float(v):.4f}")

# Also print first few val_losses for sanity
if isinstance(hist, dict) and "val_loss" in hist:
    print("\nFirst few val_loss:", [float(x) for x in hist["val_loss"][:5]])

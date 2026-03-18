import time
import tensorflow as tf
import keras
from sklearn.model_selection import StratifiedKFold


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


def make_one_fold_split(x_train, y_train, num_folds_for_split=5, random_seed=None):
    y_train_1d = y_train.reshape(-1).astype(int)
    kfold = StratifiedKFold(n_splits=num_folds_for_split, shuffle=True, random_state=random_seed)
    tr_idx, va_idx = next(kfold.split(x_train, y_train_1d))
    return (x_train[tr_idx], y_train[tr_idx]), (x_train[va_idx], y_train[va_idx]), (tr_idx, va_idx)


def find_ideal_epoch_one_fold(
    model,
    x_train, y_train,
    x_tr_split, y_tr_split,
    x_val_split, y_val_split,
    callbacks,
    fit_fn,
    fit_kwargs=None,
    epochs=10000,
    batch_size=512,
):

    kw = dict(fit_kwargs or {})
    kw.update(
        x=x_tr_split, y=y_tr_split,
        validation_data=(x_val_split, y_val_split),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    start_cpu = time.process_time()
    _ = fit_fn(model, **kw)
    end_cpu = time.process_time()

    print(f"\nCPU time spent (one-fold train): {end_cpu - start_cpu:.4f} seconds")

    new_kw = dict(fit_kwargs or {})
    new_kw.update(
        x=x_train, y=y_train,
        epochs=epochs,
        batch_size=batch_size,
    )

    print(len(x_train), len(y_train))

    start_cpu = time.process_time()
    _ = fit_fn(model, **new_kw)
    end_cpu = time.process_time()

    print(f"\nCPU time spent (one-fold train): {end_cpu - start_cpu:.4f} seconds")


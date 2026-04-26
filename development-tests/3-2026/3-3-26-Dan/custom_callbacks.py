import tensorflow as tf

class ConvergenceStopping(tf.keras.callbacks.Callback):
    """
    Stops training when the monitored metric has converged, meaning the last
    `patience` epochs stayed within `tol` of each other.

    Convergence condition:
        max(window) - min(window) <= tol
    where window = last `patience` values of the monitored metric.
    """

    def __init__(self, monitor="val_loss", tol=1e-4, patience=5, restore_best_weights=True):
        super().__init__()
        self.monitor = monitor
        self.tol = float(tol)
        self.patience = int(patience)
        self.restore_best_weights = bool(restore_best_weights)

        self.history = []
        self.best = None
        self.best_epoch = None
        self.best_weights = None
        self.stopped_epoch = None

    def on_train_begin(self, logs=None):
        self.history = []
        self.best = None
        self.best_epoch = None
        self.best_weights = None
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            return

        current = float(current)
        self.history.append(current)

        # Track best value
        if self.best is None or current < self.best:
            self.best = current
            self.best_epoch = epoch + 1
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()

        # Need enough history
        if len(self.history) < self.patience:
            return

        window = self.history[-self.patience:]
        value_range = max(window) - min(window)

        if value_range <= self.tol:
            print(
                f"\nConvergence detected: {self.monitor} varied only "
                f"{value_range:.6f} over the last {self.patience} epochs. "
                f"Stopping at epoch {epoch + 1}."
            )

            self.model.stop_training = True
            self.stopped_epoch = epoch + 1

            if self.restore_best_weights and self.best_weights is not None:
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        # Restore best weights if training finished normally
        if self.restore_best_weights and self.best_weights is not None and not self.model.stop_training:
            self.model.set_weights(self.best_weights)

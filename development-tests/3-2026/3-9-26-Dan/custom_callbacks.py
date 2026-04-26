import tensorflow as tf

class ConvergenceStopping(tf.keras.callbacks.Callback):
    """
    Stops training when the monitored metric has converged, meaning the last
    `patience` epochs stayed within `tol` of each other.

    Convergence condition:
        max(window) - min(window) <= tol
    where window = last `patience` values of the monitored metric.
    """

    def __init__(self, monitor="val_loss", tol=1e-4, patience=5, restore_best_weights=True, best_weight_identifier=None):
        super().__init__()
        self.monitor = monitor
        self.tol = float(tol)
        self.patience = int(patience)
        self.restore_best_weights = bool(restore_best_weights)

        if best_weight_identifier is None:
            self.best_weight_identifier = monitor
        else:
            self.best_weight_identifier = best_weight_identifier

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
        current_monitor_value = logs.get(self.monitor)

        if current_monitor_value is None:
            return

        current_monitor_value = float(current_monitor_value)
        self.history.append(current_monitor_value)

        best_weight_identifier_value = logs.get(self.best_weight_identifier)
        best_weight_identifier_value = float(best_weight_identifier_value)

        # Track best value
        if self.best is None or best_weight_identifier_value < self.best:
            self.best = best_weight_identifier_value
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()

        # Need enough history
        if len(self.history) < self.patience:
            return

        window = self.history[-self.patience:]
        value_range = max(window) - min(window)

        if value_range <= self.tol:

            self.model.stop_training = True
            self.stopped_epoch = epoch + 1

    def on_train_end(self, logs=None):
        # Restore best weights if training finished normally
        if self.restore_best_weights and self.best_weights is not None:
            self.model.set_weights(self.best_weights)

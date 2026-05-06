import tensorflow as tf

class DelayedEarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(self, stop_after_epoch=10, **kwargs):
        super().__init__(**kwargs)
        self.stop_after_epoch = stop_after_epoch

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        if epoch < self.stop_after_epoch:
            self.model.stop_training = False
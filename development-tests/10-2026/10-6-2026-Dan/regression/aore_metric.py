import tensorflow as tf
import numpy as np
import keras

@keras.saving.register_keras_serializable()
class AORE(tf.keras.metrics.Metric):
    def __init__(self, threshold=np.log(10), name="aore", **kwargs):
        super().__init__(name=name, **kwargs)

        self.threshold = threshold

        self.total_abs_error = self.add_weight(
            name="total_abs_error",
            initializer="zeros"
        )
        self.total_count = self.add_weight(
            name="total_count",
            initializer="zeros"
        )
        self.rare_abs_error = self.add_weight(
            name="rare_abs_error",
            initializer="zeros"
        )
        self.rare_count = self.add_weight(
            name="rare_count",
            initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        abs_error = tf.abs(y_true - y_pred)

        self.total_abs_error.assign_add(tf.reduce_sum(abs_error))
        self.total_count.assign_add(
            tf.cast(tf.size(abs_error), tf.float32)
        )

        rare_mask = tf.cast(y_true >= self.threshold, tf.float32)

        self.rare_abs_error.assign_add(
            tf.reduce_sum(abs_error * rare_mask)
        )

        self.rare_count.assign_add(
            tf.reduce_sum(rare_mask)
        )

    def result(self):
        overall_mae = self.total_abs_error / (
            self.total_count + keras.backend.epsilon()
        )

        rare_mae = self.rare_abs_error / (
            self.rare_count + keras.backend.epsilon()
        )

        return (overall_mae + rare_mae) / 2.0

    def reset_state(self):
        self.total_abs_error.assign(0.0)
        self.total_count.assign(0.0)
        self.rare_abs_error.assign(0.0)
        self.rare_count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "threshold": self.threshold,
        })
        return config
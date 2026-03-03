import tensorflow as tf
from keras.utils import register_keras_serializable

@register_keras_serializable()
class AORE(tf.keras.metrics.Metric):
    def __init__(
        self,
        name="aore",
        thresholds=(-0.5, 0.5),
        outside=True,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.square_errors = self.add_weight(name="square_errors", initializer="zeros")
        self.square_rare_errors = self.add_weight(name="square_rare_errors", initializer="zeros")

        self.count = self.add_weight(name="count", initializer="zeros")
        self.rare_count = self.add_weight(name="rare_count", initializer="zeros")

        self.thresholds = thresholds
        self.outside = outside

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), (-1,))
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), (-1,))

        if self.outside:
            rare_mask = (y_true < self.thresholds[0]) | (y_true > self.thresholds[1])
        else:
            rare_mask = (y_true > self.thresholds[0]) & (y_true < self.thresholds[1])

        errors = y_pred - y_true
        rare_errors = tf.boolean_mask(errors, rare_mask)

        self.square_errors.assign_add(tf.reduce_sum(tf.square(errors)))
        self.square_rare_errors.assign_add(tf.reduce_sum(tf.square(rare_errors)))

        self.count.assign_add(tf.cast(tf.size(errors), tf.float32))
        self.rare_count.assign_add(tf.cast(tf.size(rare_errors), tf.float32))
        # print()
        # print(tf.size(errors))
        # print(tf.size(rare_errors))
        # print()

    def result(self):
        mse_all = self.square_errors / tf.maximum(self.count, 1.0)
        mse_rare = self.square_rare_errors / tf.maximum(self.rare_count, 1.0)

        return (mse_all + mse_rare) / 2.0

    def reset_state(self):
        self.square_errors.assign(0)
        self.square_rare_errors.assign(0)
        self.count.assign(0)
        self.rare_count.assign(0)
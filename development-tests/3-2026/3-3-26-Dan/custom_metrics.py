import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package="custom_metrics")
class RareMAE(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.0, name="mae_rare", dtype=None, **kwargs):
        # NOTE: threshold has a default so deserialization won't fail even
        # if something goes wrong; but we still serialize it properly.
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.threshold = float(threshold)

        self.abs_sum = self.add_weight(name="abs_sum", initializer="zeros")
        self.n = self.add_weight(name="n", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        abs_err = tf.abs(y_true - y_pred)
        rare_mask = y_true >= tf.cast(self.threshold, tf.float32)
        rare_err = tf.boolean_mask(abs_err, rare_mask)

        self.abs_sum.assign_add(tf.reduce_sum(rare_err))
        self.n.assign_add(tf.cast(tf.size(rare_err), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.abs_sum, self.n)

    def reset_states(self):
        for v in self.variables:
            v.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config

    @classmethod
    def from_config(cls, config):
        # Keras will call this with the dict returned by get_config()
        return cls(**config)


@keras.saving.register_keras_serializable(package="custom_metrics")
class AORE(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.0, name="aore", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.threshold = float(threshold)

        self.abs_sum = self.add_weight(name="abs_sum", initializer="zeros")
        self.n = self.add_weight(name="n", initializer="zeros")
        self.rare_abs_sum = self.add_weight(name="rare_abs_sum", initializer="zeros")
        self.rare_n = self.add_weight(name="rare_n", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        abs_err = tf.abs(y_true - y_pred)

        self.abs_sum.assign_add(tf.reduce_sum(abs_err))
        self.n.assign_add(tf.cast(tf.size(abs_err), tf.float32))

        rare_mask = y_true >= tf.cast(self.threshold, tf.float32)
        rare_err = tf.boolean_mask(abs_err, rare_mask)

        self.rare_abs_sum.assign_add(tf.reduce_sum(rare_err))
        self.rare_n.assign_add(tf.cast(tf.size(rare_err), tf.float32))

    def result(self):
        overall_mae = tf.math.divide_no_nan(self.abs_sum, self.n)
        rare_mae = tf.math.divide_no_nan(self.rare_abs_sum, self.rare_n)
        return 0.5 * (overall_mae + rare_mae)

    def reset_states(self):
        for v in self.variables:
            v.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="custom_metrics")
class RegressionFalsePositives(tf.keras.metrics.Metric):

    def __init__(self, threshold, name="fp", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.threshold = float(threshold)
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

        pred_pos = y_pred >= self.threshold
        true_neg = y_true < self.threshold

        fp = tf.logical_and(pred_pos, true_neg)
        fp = tf.cast(fp, tf.float32)

        self.count.assign_add(tf.reduce_sum(fp))

    def result(self):
        return self.count

    def reset_states(self):
        self.count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


@keras.saving.register_keras_serializable(package="custom_metrics")
class RegressionFalseNegatives(tf.keras.metrics.Metric):

    def __init__(self, threshold, name="fn", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.threshold = float(threshold)
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

        pred_neg = y_pred < self.threshold
        true_pos = y_true >= self.threshold

        fn = tf.logical_and(pred_neg, true_pos)
        fn = tf.cast(fn, tf.float32)

        self.count.assign_add(tf.reduce_sum(fn))

    def result(self):
        return self.count

    def reset_states(self):
        self.count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config

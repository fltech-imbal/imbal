import tensorflow as tf
from keras.src import ops
from tensorflow import Variable, Tensor
from numpy.typing import NDArray
from imbal.metrics.util import weighted_sum

class OptimizeConfusionMetricCallback(tf.keras.callbacks.Callback):

    _enabled = tf.Variable(False, trainable=False, dtype=tf.bool)

    _true_positives = tf.Variable([0], trainable=False, dtype=tf.float32)
    _true_negatives = tf.Variable([0], trainable=False, dtype=tf.float32)
    _false_positives = tf.Variable([0], trainable=False, dtype=tf.float32)
    _false_negatives = tf.Variable([0], trainable=False, dtype=tf.float32)
    _positives = tf.Variable([0], trainable=False, dtype=tf.float32)
    _negatives = tf.Variable([0], trainable=False, dtype=tf.float32)
    _predicted_positives = tf.Variable([0], trainable=False, dtype=tf.float32)
    _predicted_negatives = tf.Variable([0], trainable=False, dtype=tf.float32)
    _sample_size = tf.Variable([0], trainable=False, dtype=tf.float32)

    _updated_this_batch = tf.Variable(False, trainable=False, dtype=tf.bool)

    @classmethod
    def is_enabled(cls) -> Variable:
        return cls._enabled

    @classmethod
    def reset_confusion_values(cls) -> None:
        cls._true_positives.assign([0])
        cls._true_negatives.assign([0])
        cls._false_positives.assign([0])
        cls._false_negatives.assign([0])
        cls._positives.assign([0])
        cls._negatives.assign([0])
        cls._predicted_positives.assign([0])
        cls._predicted_negatives.assign([0])
        cls._sample_size.assign([0])

        cls._updated_this_batch.assign(False)

    def on_train_begin(self, logs=None) -> None:
        OptimizeConfusionMetricCallback.reset_confusion_values()
        OptimizeConfusionMetricCallback._enabled.assign(True)
    def on_predict_begin(self, logs=None) -> None:
        OptimizeConfusionMetricCallback.reset_confusion_values()
        OptimizeConfusionMetricCallback._enabled.assign(True)
    def on_test_begin(self, logs=None) -> None:
        OptimizeConfusionMetricCallback.reset_confusion_values()
        OptimizeConfusionMetricCallback._enabled.assign(True)

    def on_train_end(self, logs=None) -> None:
        OptimizeConfusionMetricCallback._enabled.assign(False)
    def on_test_end(self, logs=None) -> None:
        OptimizeConfusionMetricCallback._enabled.assign(False)
    def on_predict_end(self, logs=None) -> None:
        OptimizeConfusionMetricCallback._enabled.assign(False)

    def on_epoch_begin(self, epoch, logs=None) -> None:
        OptimizeConfusionMetricCallback.reset_confusion_values()

    @classmethod
    def prepare_for_batch(cls) -> None:
        cls._updated_this_batch.assign(False)
    @classmethod
    def ensure_updated_metrics(
            cls,
            y_true,
            y_pred,
            sample_weight = None,
            dtype: type = tf.float32,
    ):

        def if_true() -> None:
            pass
        def if_false() -> None:
            cls._true_positives.assign_add(weighted_sum(y_true * y_pred, sample_weight))
            cls._sample_size.assign_add(weighted_sum(tf.ones(tf.shape(y_true), dtype=dtype), sample_weight))

            cls._positives.assign_add(weighted_sum(y_true, sample_weight))
            cls._predicted_positives.assign_add(weighted_sum(y_pred, sample_weight))

            cls._negatives.assign(ops.subtract(cls._sample_size, cls._positives))
            cls._predicted_negatives.assign(ops.subtract(cls._sample_size, cls._predicted_positives))
            cls._false_positives.assign(ops.subtract(cls._predicted_positives, cls._true_positives))
            cls._false_negatives.assign(ops.subtract(cls._positives, cls._true_positives))
            cls._true_negatives.assign(ops.subtract(cls._negatives, cls._false_positives))

            cls._updated_this_batch.assign(True)

        tf.cond(cls._updated_this_batch, if_true, if_false)


    def on_train_batch_begin(self, batch, logs=None) -> None:
        OptimizeConfusionMetricCallback.prepare_for_batch()
    def on_test_batch_begin(self, batch, logs=None) -> None:
        OptimizeConfusionMetricCallback.prepare_for_batch()
    def on_predict_batch_begin(self, batch, logs=None) -> None:
        OptimizeConfusionMetricCallback.prepare_for_batch()

    @classmethod
    def tp(cls) -> Variable:
        return cls._true_positives
    @classmethod
    def tn(cls) -> Variable:
        return cls._true_negatives
    @classmethod
    def fp(cls) -> Variable:
        return cls._false_positives
    @classmethod
    def fn(cls) -> Variable:
        return cls._false_negatives
    @classmethod
    def pos(cls) -> Variable:
        return cls._positives
    @classmethod
    def neg(cls) -> Variable:
        return cls._negatives
    @classmethod
    def ppos(cls) -> Variable:
        return cls._predicted_positives
    @classmethod
    def pneg(cls) -> Variable:
        return cls._predicted_negatives
    @classmethod
    def ss(cls) -> Variable:
        return cls._sample_size





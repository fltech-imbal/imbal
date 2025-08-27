import tensorflow as tf
from numpy.typing import NDArray

class FPR(tf.keras.Metric):
    def __init__(self,
                 threshold : float = 0.5,
                 name='fpr',
                 **kwargs) -> None:
        super(FPR, self).__init__(name=name, **kwargs)
        self._threshold = threshold

        self.fp = self.add_weight(name="fp", initializer="zeros", dtype=tf.float32)
        self.neg = self.add_weight(name="neg", initializer="zeros", dtype=tf.float32)

    def update_state(self,
                     y_true : NDArray,
                     y_pred : NDArray,
                     sample_weight = None) -> None:
        y_pred = tf.cast(y_pred > self._threshold, tf.float32)
        y_true = tf.reshape(tf.cast(y_true, tf.float32), (-1, 1))

        fp = tf.reduce_sum((1 - y_true) * y_pred)
        neg = tf.reduce_sum(1 - y_true)

        self.fp.assign_add(fp)
        self.neg.assign_add(neg)

    def result(self) -> float:
        return tf.math.divide_no_nan(self.fp, self.neg)

    def reset_states(self) -> None:
        self.fp.assign(0.0)
        self.neg.assign(0.0)
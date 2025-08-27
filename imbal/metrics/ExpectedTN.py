import tensorflow as tf
from typing import List
from numpy.typing import NDArray

class ExpectedTN(tf.keras.Metric):
    def __init__(self,
                 threshold : float = 0.5,
                 name='expected_tn',
                 **kwargs) -> None:
        super(ExpectedTN, self).__init__(name=name, **kwargs)
        self._threshold = threshold

        self.neg = self.add_weight(name="neg", initializer="zeros", dtype=tf.float32)
        self.pneg = self.add_weight(name="pneg", initializer="zeros", dtype=tf.float32)
        self.sample_size = self.add_weight(name="sample_size", initializer="zeros", dtype=tf.float32)

    def update_state(self,
                     y_true : List | NDArray,
                     y_pred : List | NDArray,
                     sample_weight = None) -> None:
        y_pred = tf.cast(y_pred > self._threshold, tf.int32)
        y_true = tf.reshape(tf.cast(y_true, tf.float32), (-1, 1))

        pneg = tf.reduce_sum(1 - y_pred)
        neg = tf.reduce_sum(1 - y_true)
        sample_size = tf.size(y_true)

        self.neg.assign_add(neg)
        self.pneg.assign_add(pneg)
        self.sample_size.assign_add(sample_size)

    def result(self) -> float:
        return tf.math.divide_no_nan(self.neg * self.pneg, self.sample_size)

    def reset_states(self) -> None:
        self.neg.assign(0.0)
        self.pneg.assign(0.0)
        self.sample_size.assign(0.0)
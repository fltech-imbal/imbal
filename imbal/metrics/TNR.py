import tensorflow as tf
from numpy.typing import NDArray

class TNR(tf.keras.Metric):
    def __init__(self,
                 threshold : float = 0.5,
                 name='tnr',
                 **kwargs) -> None:
        super(TNR, self).__init__(name=name, **kwargs)
        self._threshold = threshold

        self.tn = self.add_weight(name="tn", initializer="zeros", dtype=tf.float32)
        self.neg = self.add_weight(name="neg", initializer="zeros", dtype=tf.float32)

    def update_state(self,
                     y_true : NDArray,
                     y_pred : NDArray,
                     sample_weight = None) -> None:
        y_pred = tf.cast(y_pred > self._threshold, tf.float32)
        y_true = tf.reshape(tf.cast(y_true, tf.float32), (-1, 1))

        tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
        neg = tf.reduce_sum(1 - y_true)

        self.tn.assign_add(tn)
        self.neg.assign_add(neg)

    def result(self) -> float:
        return tf.math.divide_no_nan(self.tn, self.neg)

    def reset_states(self) -> None:
        self.tn.assign(0.0)
        self.neg.assign(0.0)
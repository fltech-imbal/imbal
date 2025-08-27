import tensorflow as tf
from typing import List
from numpy.typing import NDArray

class TPR(tf.keras.Metric):
    def __init__(self,
                 threshold : float = 0.5,
                 name='tpr',
                 **kwargs) -> None:
        super(TPR, self).__init__(name=name, **kwargs)
        self._threshold = threshold

        self.tp = self.add_weight(name="tp", initializer="zeros", dtype=tf.float32)
        self.pos = self.add_weight(name="pos", initializer="zeros", dtype=tf.float32)

    def update_state(self,
                     y_true : List | NDArray,
                     y_pred : List | NDArray,
                     sample_weight = None) -> None:
        y_pred = tf.cast(y_pred > self._threshold, tf.float32)
        y_true = tf.reshape(tf.cast(y_true, tf.float32), (-1, 1))

        tp = tf.reduce_sum(y_true * y_pred)
        pos = tf.reduce_sum(y_true)

        self.tp.assign_add(tp)
        self.pos.assign_add(pos)

    def result(self) -> float:
        return tf.math.divide_no_nan(self.tp, self.pos)

    def reset_states(self) -> None:
        self.tp.assign(0.0)
        self.pos.assign(0.0)
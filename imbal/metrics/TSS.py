import tensorflow as tf
from typing import List
from numpy.typing import NDArray

class TSS(tf.keras.Metric):
    def __init__(self,
                 threshold : float = 0.5,
                 name='tss',
                 **kwargs) -> None:
        super(TSS, self).__init__(name=name, **kwargs)
        self._threshold = threshold

        self.pos = self.add_weight(name="pos", initializer="zeros", dtype=tf.float32)
        self.tp = self.add_weight(name="tp", initializer="zeros", dtype=tf.float32)
        self.neg = self.add_weight(name="neg", initializer="zeros", dtype=tf.float32)
        self.fp = self.add_weight(name="fp", initializer="zeros", dtype=tf.float32)
        self.sample_size = self.add_weight(name="sample_size", initializer="zeros", dtype=tf.float32)

    def update_state(self,
                     y_true : List | NDArray,
                     y_pred : List | NDArray,
                     sample_weight = None) -> None:
        y_pred = tf.cast(y_pred > self._threshold, tf.float32)
        y_true = tf.reshape(tf.cast(y_true, tf.float32), (-1, 1))

        neg = tf.reduce_sum(1 - y_true)
        pos = tf.reduce_sum(y_true)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)

        self.pos.assign_add(pos)
        self.neg.assign_add(neg)
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)

    def result(self) -> float:
        return tf.math.divide_no_nan(self.tp, self.pos) - tf.math.divide_no_nan(self.fp, self.neg)

    def reset_states(self) -> None:
        self.neg.assign(0.0)
        self.pos.assign(0.0)
        self.fp.assign(0.0)
        self.tp.assign(0.0)
        self.sample_size.assign(0.0)
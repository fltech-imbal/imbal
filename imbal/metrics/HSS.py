import tensorflow as tf
from typing import List
from numpy.typing import NDArray

class HSS(tf.keras.Metric):
    def __init__(self,
                 threshold : float = 0.5,
                 name='hss',
                 **kwargs) -> None:
        super(HSS, self).__init__(name=name, **kwargs)
        self._threshold = threshold

        self.tp = self.add_weight(name="tp", initializer="zeros", dtype=tf.float32)
        self.fp = self.add_weight(name="fp", initializer="zeros", dtype=tf.float32)
        self.tn = self.add_weight(name="tn", initializer="zeros", dtype=tf.float32)
        self.fn = self.add_weight(name="fn", initializer="zeros", dtype=tf.float32)
        self.sample_size = self.add_weight(name="sample_size", initializer="zeros", dtype=tf.float32)

    def update_state(self,
                     y_true : List | NDArray,
                     y_pred : List | NDArray,
                     sample_weight = None) -> None:
        y_pred = tf.cast(y_pred > self._threshold, tf.float32)
        y_true = tf.reshape(tf.cast(y_true, tf.float32), (-1, 1))

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        sample_size = tf.size(y_true)

        self.fn.assign_add(fn)
        self.tn.assign_add(tn)
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.sample_size.assign_add(sample_size)

    def result(self) -> float:
        expected_correct = tf.math.divide_no_nan((self.tp + self.fn) * (self.tp + self.fp),
                                                 self.sample_size) + tf.math.divide_no_nan(
            (self.tn + self.fp) * (self.tn + self.fn), self.sample_size)
        return (self.tp + self.tn - expected_correct) / (self.sample_size - expected_correct)

    def reset_states(self) -> None:
        self.tn.assign(0.0)
        self.fn.assign(0.0)
        self.fp.assign(0.0)
        self.tp.assign(0.0)
        self.sample_size.assign(0.0)
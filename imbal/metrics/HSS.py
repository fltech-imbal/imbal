import tensorflow as tf
from typing import List
from numpy.typing import NDArray
from imbal.metrics import ConfusionMatrix
from imbal.metrics.util import expected_correct
from keras.src import ops

class HSS(tf.keras.Metric):
    def __init__(self,
                 threshold : float = 0.5) -> None:
        super(HSS, self).__init__()
        self.name = 'hss'
        self._confusion_matrix : ConfusionMatrix | None = None
        self._threshold = threshold

    def update_state(self,
                     y_true : List | NDArray,
                     y_pred : List | NDArray) -> None:
        self._confusion_matrix = ConfusionMatrix(y_true, y_pred, self._threshold)

    def result(self) -> float:
        if self._confusion_matrix is None:
            raise ValueError('update_state() must be called before calling result()')

        ec = expected_correct(self._confusion_matrix)
        return tf.divide(ops.cast(self._confusion_matrix.tp() + self._confusion_matrix.tn(), tf.float64) - ec, ops.cast(self._confusion_matrix.sample_size() - ec, tf.float64))
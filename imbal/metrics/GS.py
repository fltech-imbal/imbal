import tensorflow as tf
from typing import List
from numpy.typing import NDArray
from imbal.metrics import ConfusionMatrix
from imbal.metrics.util import expected_tp
from keras.src import ops

class GS(tf.keras.Metric):
    def __init__(self,
                 threshold : float = 0.5) -> None:
        super(GS, self).__init__()
        self.name = 'gs'
        self._confusion_matrix : ConfusionMatrix | None = None
        self._threshold = threshold

    def update_state(self,
                     y_true : List | NDArray,
                     y_pred : List | NDArray) -> None:
        self._confusion_matrix = ConfusionMatrix(y_true, y_pred, self._threshold)

    def result(self) -> float:
        if self._confusion_matrix is None:
            raise ValueError('update_state() must be called before calling result()')

        ex_tp = expected_tp(self._confusion_matrix)
        numerator = ops.cast(self._confusion_matrix.tp(), tf.float64) - ex_tp
        denominator = ops.cast(self._confusion_matrix.ppos() + self._confusion_matrix.fn(), tf.float64) - ex_tp
        return numerator / denominator
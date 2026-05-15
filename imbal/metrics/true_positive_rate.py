from numpy.typing import NDArray
from typing import Tuple
from tensorflow import Tensor

from imbal.metrics.util import ConfusionMatrixMetric
from imbal.metrics.confusion_matrix import ConfusionMatrix, ConfusionMatrixData
from keras.src import ops

class TruePositiveRate(ConfusionMatrixMetric):
    """
     imbal.metrics.TruePositiveRate
    ========================================
    Handles True Positive Rate
    """
    def __init__(
        self,
        threshold : None | float = 0.5,
        name : str = 'true_positive_rate',
        dtype : type | None = None
    ) -> None:

        super().__init__(
            name=name,
            dtype=dtype,
            threshold=threshold
        )

        self._true_positives = None
        self._positives = None
        self._direction = 'up'

    def _build(
        self,
        y_true_shape : Tuple,
        y_pred_shape : Tuple
    ) -> None:
        super()._build(y_true_shape, y_pred_shape)

        self._true_positives = super()._add_zeros_variable("true_positives")
        self._positives = super()._add_zeros_variable("positives")
        self._built = True

    def _complete_update(
            self,
            y_true: NDArray | Tensor,
            y_pred: NDArray | Tensor,
            sample_weight: NDArray | Tensor | None = None
    ):
        ConfusionMatrix.compute({
            ConfusionMatrixData.TRUE_POSITIVE: self._true_positives,
            ConfusionMatrixData.POSITIVE: self._positives,
        }, y_true, y_pred, sample_weight)

    def result(self) -> Tensor:
        return ops.divide(self._true_positives, self._positives)

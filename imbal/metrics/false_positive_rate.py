from numpy.typing import NDArray
from typing import Tuple
from tensorflow import Tensor
import tensorflow as tf
from imbal.metrics.util import ConfusionMatrixMetric, weighted_sum

class FalsePositiveRate(ConfusionMatrixMetric):
    def __init__(
        self,
        threshold : None | float = 0.5,
        name : str = 'false_positive_rate',
        dtype : type | None = None
    ) -> None:

        super().__init__(
            name=name,
            dtype=dtype,
            threshold=threshold
        )

        self._false_positives = None
        self._negatives = None

    def _build(
        self,
        y_true_shape : Tuple,
        y_pred_shape : Tuple
    ) -> None:
        super()._build(y_true_shape, y_pred_shape)

        self._false_positives = super()._add_zeros_variable("false_positives")
        self._negatives = super()._add_zeros_variable("negatives")
        self._built = True

    def _complete_update(
            self,
            y_true: NDArray | Tensor,
            y_pred: NDArray | Tensor,
            sample_weight: NDArray | Tensor | None = None
    ):
        self._false_positives.assign_add(weighted_sum((1 - y_true) * y_pred, sample_weight))
        self._negatives.assign_add(weighted_sum(1 - y_true, sample_weight))

    def result(self) -> Tensor:
        return self._false_positives / self._negatives

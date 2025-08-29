from numpy.typing import NDArray
from typing import Tuple
from tensorflow import Tensor
from imbal.metrics.ConfusionMatrixMetric import ConfusionMatrixMetric
from imbal.metrics.util import weighted_sum

class TrueNegativeRate(ConfusionMatrixMetric):
    def __init__(
        self,
        threshold : None | float = 0.5,
        name : str = 'true_negative_rate',
        dtype : type | None = None
    ) -> None:

        super().__init__(
            name=name,
            dtype=dtype,
            threshold=threshold
        )

        self._true_negatives = None
        self._negatives = None

    def build(
        self,
        y_true_shape : Tuple,
        y_pred_shape : Tuple
    ) -> None:
        super().build(y_true_shape, y_pred_shape)

        self._true_negatives = super().add_zeros_variable("true_negatives")
        self._negatives = super().add_zeros_variable("negatives")
        self._built = True

    def complete_update(
            self,
            y_true: NDArray | Tensor,
            y_pred: NDArray | Tensor,
            sample_weight: NDArray | Tensor | None = None
    ):
        self._true_negatives.assign_add(weighted_sum((1 - y_true) * (1 - y_pred), sample_weight))
        self._negatives.assign_add(weighted_sum(1 - y_true, sample_weight))

    def result(self) -> Tensor:
        return self._true_negatives / self._negatives

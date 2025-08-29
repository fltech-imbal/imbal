from numpy.typing import NDArray
from typing import Tuple
from tensorflow import Tensor
from imbal.metrics.ConfusionMatrixMetric import ConfusionMatrixMetric
from imbal.metrics.util import weighted_sum

class TrueSkillStatistic(ConfusionMatrixMetric):
    def __init__(
        self,
        threshold : None | float = 0.5,
        name : str = 'true_skill_statistic',
        dtype : type | None = None
    ) -> None:

        super().__init__(
            name=name,
            dtype=dtype,
            threshold=threshold
        )

        self._true_positives = None
        self._positives = None
        self._false_positives = None
        self._negatives = None

    def build(
        self,
        y_true_shape : Tuple,
        y_pred_shape : Tuple
    ) -> None:
        super().build(y_true_shape, y_pred_shape)

        self._true_positives = super().add_zeros_variable("true_positives")
        self._positives = super().add_zeros_variable("positives")
        self._false_positives = super().add_zeros_variable("false_positives")
        self._negatives = super().add_zeros_variable("negatives")
        self._built = True

    def complete_update(
            self,
            y_true: NDArray | Tensor,
            y_pred: NDArray | Tensor,
            sample_weight: NDArray | Tensor | None = None
    ):
        self._true_positives.assign_add(weighted_sum(y_true * y_pred, sample_weight))
        self._positives.assign_add(weighted_sum(y_true, sample_weight))
        self._false_positives.assign_add(weighted_sum((1 - y_true) * y_pred, sample_weight))
        self._negatives.assign_add(weighted_sum(1 - y_true, sample_weight))

    def result(self) -> Tensor:
        return self._true_positives / self._positives - self._false_positives / self._negatives

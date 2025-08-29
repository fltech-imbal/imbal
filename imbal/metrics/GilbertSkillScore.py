from numpy.typing import NDArray
from typing import Tuple
from tensorflow import Tensor
from imbal.metrics.ConfusionMatrixMetric import ConfusionMatrixMetric
from imbal.metrics.util import weighted_sum
import tensorflow as tf

class GilbertSkillScore(ConfusionMatrixMetric):
    def __init__(
        self,
        threshold : None | float = 0.5,
        name : str = 'gilbert_skill_score',
        dtype : type | None = None
    ) -> None:

        super().__init__(
            name=name,
            dtype=dtype,
            threshold=threshold
        )

        self._true_positive = None
        self._positive = None
        self._false_positive = None
        self._false_negative = None
        self._sample_size = None

    def build(
        self,
        y_true_shape : Tuple,
        y_pred_shape : Tuple
    ) -> None:
        super().build(y_true_shape, y_pred_shape)

        self._positive = super().add_zeros_variable("positive")
        self._true_positive = super().add_zeros_variable("true_positive")
        self._false_positive = super().add_zeros_variable("false_positive")
        self._false_negative = super().add_zeros_variable("false_negative")
        self._sample_size = super().add_zeros_variable("sample_size")
        self._built = True

    def complete_update(
            self,
            y_true: NDArray | Tensor,
            y_pred: NDArray | Tensor,
            sample_weight: NDArray | Tensor | None = None
    ):

        self._positive.assign_add(weighted_sum(y_true, sample_weight))
        self._true_positive.assign_add(weighted_sum(y_true * y_pred, sample_weight))
        self._false_positive.assign_add(weighted_sum((1 - y_true) * y_pred, sample_weight))
        self._false_negative.assign_add(weighted_sum(y_true * (1 - y_pred), sample_weight))
        self._sample_size.assign_add(weighted_sum(tf.ones(tf.shape(y_true), dtype=self.dtype), sample_weight))

    def result(self) -> Tensor:
        tpr = self._positive * (self._true_positive + self._false_positive) / self._sample_size
        return (self._true_positive - tpr) / (self._true_positive + self._false_positive + self._false_negative - tpr)

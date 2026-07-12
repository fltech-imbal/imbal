from numpy.typing import NDArray
from typing import Tuple
from tensorflow import Tensor
from imbal.metrics.util import ConfusionMatrixMetric, weighted_sum
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class ExpectedTruePositive(ConfusionMatrixMetric):
    def __init__(
        self,
        threshold : None | float = 0.5,
        name : str = 'expected_true_positive',
        dtype : type | None = None
    ) -> None:

        super().__init__(
            name=name,
            dtype=dtype,
            threshold=threshold
        )

        self._positive = None
        self._predicted_positive = None
        self._sample_size = None
        self._direction = 'up'

    def _build(
        self,
        y_true_shape : Tuple,
        y_pred_shape : Tuple
    ) -> None:
        super()._build(y_true_shape, y_pred_shape)

        self._positive = super()._add_zeros_variable("positive")
        self._predicted_positive = super()._add_zeros_variable("predicted_positive")
        self._sample_size = super()._add_zeros_variable("sample_size")
        self._built = True

    def _complete_update(
            self,
            y_true: NDArray | Tensor,
            y_pred: NDArray | Tensor,
            sample_weight: NDArray | Tensor | None = None
    ):
        self._positive.assign_add(weighted_sum(y_true, sample_weight))
        self._predicted_positive.assign_add(weighted_sum(y_pred, sample_weight))
        self._sample_size.assign_add(weighted_sum(tf.ones(tf.shape(y_true), dtype=self.dtype), sample_weight))

    def result(self) -> Tensor:
        return self._positive * self._predicted_positive /  self._sample_size

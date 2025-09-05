from numpy.typing import NDArray
from typing import Tuple
from tensorflow import Tensor
from imbal.metrics.util import ConfusionMatrixMetric, weighted_sum
from imbal.metrics.optimize_confusion_metric_callback import OptimizeConfusionMetricCallback as ocmc
import tensorflow as tf

class ExpectedCorrect(ConfusionMatrixMetric):
    def __init__(
        self,
        threshold : None | float = 0.5,
        name : str = 'expected_correct',
        dtype : type | None = None
    ) -> None:

        super().__init__(
            name=name,
            dtype=dtype,
            threshold=threshold
        )

        self._positive = None
        self._predicted_positive = None
        self._negative = None
        self._predicted_negative = None
        self._sample_size = None

    def _build(
        self,
        y_true_shape : Tuple,
        y_pred_shape : Tuple
    ) -> None:
        super()._build(y_true_shape, y_pred_shape)

        self._negative = super()._add_zeros_variable("negative")
        self._predicted_negative = super()._add_zeros_variable("predicted_negative")
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
        def optimized_update() -> None:
            ocmc.ensure_updated_metrics(y_true, y_pred, sample_weight)
            self._negative.assign(ocmc.neg())
            self._predicted_negative.assign(ocmc.pneg())
            self._positive.assign(ocmc.pos())
            self._predicted_positive.assign(ocmc.ppos())
            self._sample_size.assign(ocmc.ss())
        def manual_update() -> None:
            self._negative.assign_add(weighted_sum(1 - y_true, sample_weight))
            self._predicted_negative.assign_add(weighted_sum(1 - y_pred, sample_weight))
            self._positive.assign_add(weighted_sum(y_true, sample_weight))
            self._predicted_positive.assign_add(weighted_sum(y_pred, sample_weight))
            self._sample_size.assign_add(weighted_sum(tf.ones(tf.shape(y_true), dtype=self.dtype), sample_weight))

        tf.cond(ocmc.is_enabled(), optimized_update, manual_update)

    def result(self) -> Tensor:
        return (self._negative * self._predicted_negative /  self._sample_size +
                self._positive * self._predicted_positive /  self._sample_size)

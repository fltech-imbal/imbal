from numpy.typing import NDArray
from typing import Tuple
from tensorflow import Tensor
import tensorflow as tf
from imbal.metrics.util import ConfusionMatrixMetric, weighted_sum
from imbal.metrics.optimize_confusion_metric_callback import OptimizeConfusionMetricCallback as ocmc

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

    def _build(
        self,
        y_true_shape : Tuple,
        y_pred_shape : Tuple
    ) -> None:
        super()._build(y_true_shape, y_pred_shape)

        self._true_negatives = super()._add_zeros_variable("true_negatives")
        self._negatives = super()._add_zeros_variable("negatives")
        self._built = True

    def _complete_update(
            self,
            y_true: NDArray | Tensor,
            y_pred: NDArray | Tensor,
            sample_weight: NDArray | Tensor | None = None
    ):
        def optimized_update() -> None:
            ocmc.ensure_updated_metrics(y_true, y_pred, sample_weight)
            self._true_negatives.assign(ocmc.tn())
            self._negatives.assign(ocmc.neg())
        def manual_update() -> None:
            self._true_negatives.assign_add(weighted_sum((1 - y_true) * (1 - y_pred), sample_weight))
            self._negatives.assign_add(weighted_sum(1 - y_true, sample_weight))

        tf.cond(ocmc.is_enabled(), optimized_update, manual_update)

    def result(self) -> Tensor:
        return self._true_negatives / self._negatives

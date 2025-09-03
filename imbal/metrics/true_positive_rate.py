from numpy.typing import NDArray
from typing import Tuple
from tensorflow import Tensor
import tensorflow as tf

from imbal.metrics.util import ConfusionMatrixMetric, weighted_sum
from imbal.metrics.confusion_matrix import ConfusionMatrix, ConfusionMatrixData
from imbal.metrics.optimize_confusion_metric_callback import OptimizeConfusionMetricCallback as ocmc
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

    def _build(
        self,
        y_true_shape : Tuple,
        y_pred_shape : Tuple
    ) -> None:
        """
        This is a test. Build function, hello!
        """
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

        def optimized_update() -> None:
            ocmc.ensure_updated_metrics(y_true, y_pred, sample_weight)
            self._true_positives.assign(ocmc.tp())
            self._positives.assign(ocmc.pos())
        def manual_update() -> None:
            ConfusionMatrix.compute({
                ConfusionMatrixData.TRUE_POSITIVE: self._true_positives,
                ConfusionMatrixData.POSITIVE: self._positives,
            }, y_true, y_pred, sample_weight)

        tf.cond(ocmc.is_enabled(), optimized_update, manual_update)




    def result(self) -> Tensor:
        return ops.divide(self._true_positives, self._positives)

from numpy.typing import NDArray
from typing import Tuple
from tensorflow import Tensor
import tensorflow as tf
from imbal.metrics.util import ConfusionMatrixMetric, weighted_sum
from imbal.metrics.optimize_confusion_metric_callback import OptimizeConfusionMetricCallback as ocmc

class CriticalSuccessIndex(ConfusionMatrixMetric):
    """
    Computes the Critical Success Index.

    Formula:

    .. code-block:: python

       critical_success_index = true_positive / (true_positive + false_positive + false_negative)

    This quotient the "skill" of a system. The output range is :code:`[0, 1]`.
    An output of :code:`0` means the system is entirely unskilled (guessing
    randomly), while an output of :code:`1` means the system is entirely
    skilled (guessing perfectly).

    For use in TensorFlow's :code:`model.compile` function, this class
    can be passed as a metric, along with any of the following string type
    aliases:

    * :code:`"CriticalSuccessIndex"`
    * :code:`"critical_success_index"`
    * :code:`"csi"`
    * :code:`"CSI"`
    * :code:`"threat_score"`

    Args:
        threshold : Optional, default :code:`0.5`. The value which a given
            prediction must be above in order to be considered a positive
            guess. All predictions below or equal to this threshold will be
            considered a negative guess.
        name : Optional, default :code:`"critical_success_index"`. String name
            of the metric instance.
        dtype : Optional, default :code:`None`. Data type of the metric result.

    Returns:
        float: Critical success index.

    Example:

    .. code-block:: python

        metric = imbal.metrics.CriticalSuccessIndex(threshold=0.5)
        y_true = np.array([[1,1,1], [1,0,0], [1,1,0]], np.int32)
        y_pred = np.array([[0.2,0.6,0.7],[0.2,0.6,0.6],[0.6,0.8,0.0]], np.float32)
        metric.update_state(y_true, y_pred)
        result = metric.result()
    """
    def __init__(
        self,
        threshold = 0.5,
        name = 'critical_success_index',
        dtype = None
    ) -> None:

        super().__init__(
            name=name,
            dtype=dtype,
            threshold=threshold
        )

        self._true_positive = None
        self._false_positive = None
        self._false_negative = None

    def build(
        self,
        y_true_shape : Tuple,
        y_pred_shape : Tuple
    ) -> None:
        super()._build(y_true_shape, y_pred_shape)

        self._true_positive = super()._add_zeros_variable("true_positive")
        self._false_positive = super()._add_zeros_variable("false_positive")
        self._false_negative = super()._add_zeros_variable("false_negative")
        self._built = True

    def _complete_update(
            self,
            y_true: NDArray | Tensor,
            y_pred: NDArray | Tensor,
            sample_weight: NDArray | Tensor | None = None
    ):
        def optimized_update() -> None:
            ocmc.ensure_updated_metrics(y_true, y_pred, sample_weight)
            self._true_positive.assign(ocmc.tp())
            self._false_positive.assign(ocmc.fp())
            self._false_negative.assign(ocmc.fn())
        def manual_update() -> None:
            self._true_positive.assign_add(weighted_sum(y_true * y_pred, sample_weight))
            self._false_positive.assign_add(weighted_sum((1 - y_true) * y_pred, sample_weight))
            self._false_negative.assign_add(weighted_sum(y_true * (1 - y_pred), sample_weight))

        tf.cond(ocmc.is_enabled(), optimized_update, manual_update)

    def result(self) -> Tensor:
        return self._true_positive / (self._true_positive + self._false_positive + self._false_negative)

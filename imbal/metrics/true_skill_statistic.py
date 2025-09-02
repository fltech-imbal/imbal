from numpy.typing import NDArray
from typing import Tuple
from tensorflow import Tensor
import tensorflow as tf
from imbal.metrics.util import ConfusionMatrixMetric, weighted_sum
from imbal.metrics.optimize_confusion_metric_callback import OptimizeConfusionMetricCallback as ocmc

class TrueSkillStatistic(ConfusionMatrixMetric):
    """
    Computes the True Skill Statistic.

    Formula:

    .. code-block:: python

       true_skill_statistic = true_positive_rate - false_positive_rate

    This difference represents the "skill" of a system. Assuming a system
    will not perform worse than random, the output range is :code:`[0, 1]`.
    An output of :code:`0` means the system is entirely unskilled (guessing
    randomly), while an output of :code:`1` means the system is entirely
    skilled (guessing perfectly).

    For use in TensorFlow's :code:`model.compile` function, this class
    can be passed as a metric, along with any of the following string type
    aliases:

    * :code:`"TrueSkillStatistic"`
    * :code:`"true_skill_statistic"`
    * :code:`"tss"`
    * :code:`"TSS"`
    * :code:`"j_statistic"`
    * :code:`"youdens_index"`

    Args:
        threshold : Optional, default :code:`0.5`. The value which a given
            prediction must be above in order to be considered a positive
            guess. All predictions below or equal to this threshold will be
            considered a negative guess.
        name : Optional, default :code:`"true_skill_statistic"`. String name
            of the metric instance.
        dtype : Optional, default :code:`None`. Data type of the metric result.

    Returns:
        float: True skill statistic.

    Example:

    .. code-block:: python

        metric = imbal.metrics.TrueSkillStatistic(threshold=0.5)
        y_true = np.array([[1,1,1], [1,0,0], [1,1,0]], np.int32)
        y_pred = np.array([[0.2,0.6,0.7],[0.2,0.6,0.6],[0.6,0.8,0.0]], np.float32)
        metric.update_state(y_true, y_pred)
        result = metric.result()
    """
    def __init__(
        self,
        threshold = 0.5,
        name = 'true_skill_statistic',
        dtype = None
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

    def _build(
        self,
        y_true_shape : Tuple,
        y_pred_shape : Tuple
    ) -> None:
        super()._build(y_true_shape, y_pred_shape)

        self._true_positives = super()._add_zeros_variable("true_positives")
        self._positives = super()._add_zeros_variable("positives")
        self._false_positives = super()._add_zeros_variable("false_positives")
        self._negatives = super()._add_zeros_variable("negatives")
        self._built = True

    def _complete_update(
            self,
            y_true: NDArray | Tensor,
            y_pred: NDArray | Tensor,
            sample_weight: NDArray | Tensor | None = None
    ):
        """
        For internal class use only. Updates the confusion metric, with
        options for calling upon pre-computed confusion matrix values
        """
        def optimized_update() -> None:
            ocmc.ensure_updated_metrics(y_true, y_pred, sample_weight)
            self._true_positives.assign(ocmc.tp())
            self._positives.assign(ocmc.pos())
            self._false_positives.assign(ocmc.fp())
            self._negatives.assign(ocmc.neg())
        def manual_update() -> None:
            self._true_positives.assign_add(weighted_sum(y_true * y_pred, sample_weight))
            self._positives.assign_add(weighted_sum(y_true, sample_weight))
            self._false_positives.assign_add(weighted_sum((1 - y_true) * y_pred, sample_weight))
            self._negatives.assign_add(weighted_sum(1 - y_true, sample_weight))

        tf.cond(ocmc.is_enabled(), optimized_update, manual_update)

    def result(self) -> Tensor:
        return self._true_positives / self._positives - self._false_positives / self._negatives

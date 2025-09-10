from numpy.typing import NDArray
from typing import Tuple
from tensorflow import Tensor
import tensorflow as tf
from imbal.metrics.util import ConfusionMatrixMetric, weighted_sum
from imbal.experimental.optimize_confusion_metric_callback import OptimizeConfusionMetricCallback as ocmc
from keras.src.metrics import metrics_utils

class TrueSkillStatistic(ConfusionMatrixMetric):
    r"""
    Computes the True Skill Statistic.

    Formula:

    .. math::

       \text{True Skill Statistic} = true\_positive\_rate - false\_positive\_rate

    Here you can find more information about :doc:`true positive rate </imbal/metrics/submetrics/true_positive_rate>`
    and :doc:`false positive rate </imbal/metrics/submetrics/false_positive_rate>`.

    This difference represents the "skill" of a system. Assuming a system
    will not perform worse than random, the output range is :code:`[0, 1]`.
    An output of :code:`0` means the system is entirely unskilled (guessing
    randomly), while an output of :code:`1` means the system is entirely
    skilled (guessing perfectly).

    Example usage:

    .. code-block:: python

        metric = imbal.metrics.TrueSkillStatistic(threshold=0.5)
        y_true = np.array([[1,1,1], [1,0,0], [1,1,0]], np.int32)
        y_pred = np.array([[0.2,0.6,0.7],[0.2,0.6,0.6],[0.6,0.8,0.0]], np.float32)
        metric.update_state(y_true, y_pred)
        result = metric.result()

    For use in TensorFlow's :code:`model.compile` function, this class
    can be passed as a class instance or as any of the following string type
    aliases:

    * :code:`"TrueSkillStatistic"`
    * :code:`"true_skill_statistic"`
    * :code:`"tss"`
    * :code:`"TSS"`

    Example:

    .. code-block:: python

       model.compile(
           optimizer="adam",
           loss="binary_crossentropy",
           metrics=["tss"]
       )

    The True Skill Statistic is equal to the :doc:`J Statistic</imbal/metrics/j_statistic>`
    and :doc:`Youden's Index</imbal/metrics/youdens_index>`.

    **Note:** Where appropriate, documentation for functions from :code:`tf.keras.Metric` has been
    overridden to be more descriptive. Any other non-descriptive documentation of individual functions
    on this page is due to a lack of documentation in TensorFlow's original source code. Still, TensorFlow's
    documentation and source code for the :code:`Metric` class can be found `here <https://www.tensorflow.org/api_docs/python/tf/keras/Metric>`_.

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
            metrics_utils.update_confusion_matrix_variables(
                {
                    metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self._true_positives,
                    metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self._false_positives
                },
                y_true,
                y_pred,
                metrics_utils.parse_init_thresholds(None, self._threshold),
                sample_weight=sample_weight
            )

            self._positives.assign_add(weighted_sum(y_true, sample_weight))
            self._negatives.assign_add(weighted_sum(1 - y_true, sample_weight))

        tf.cond(ocmc.is_enabled(), optimized_update, manual_update)

    def result(self) -> Tensor:
        """
        Computes the current value of the metric based on the accumulated data.

        Returns:
            The true skill statistic of the accumulated data.
        """
        return self._true_positives / self._positives - self._false_positives / self._negatives

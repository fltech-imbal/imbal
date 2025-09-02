from numpy.typing import NDArray
from typing import Tuple
from tensorflow import Tensor
from imbal.metrics.util import ConfusionMatrixMetric, weighted_sum
from imbal.metrics.optimize_confusion_metric_callback import OptimizeConfusionMetricCallback as ocmc
import tensorflow as tf

class HeikdeSkillScore(ConfusionMatrixMetric):
    """
    Computes the Heikde Skill Score.

    Formula:

    .. code-block:: python

       heikde_skill_score = (true_positive + true_negative - expected_correct) / (sample_size - expected_correct)

    This quotient the "skill" of a system. Assuming a system
    will not perform worse than random, the output range is :code:`[0, 1]`.
    An output of :code:`0` means the system is entirely unskilled (guessing
    randomly), while an output of :code:`1` means the system is entirely
    skilled (guessing perfectly).

    For use in TensorFlow's :code:`model.compile` function, this class
    can be passed as a metric, along with any of the following string type
    aliases:

    * :code:`"HeikdeSkillScore"`
    * :code:`"heikde_skill_score"`
    * :code:`"hss"`
    * :code:`"HSS"`

    Args:
        threshold : Optional, default :code:`0.5`. The value which a given
            prediction must be above in order to be considered a positive
            guess. All predictions below or equal to this threshold will be
            considered a negative guess.
        name : Optional, default :code:`"heikde_skill_score"`. String name
            of the metric instance.
        dtype : Optional, default :code:`None`. Data type of the metric result.

    Returns:
        float: Heikde skill score.

    Example:

    .. code-block:: python

        metric = imbal.metrics.HeikdeSkillScore(threshold=0.5)
        y_true = np.array([[1,1,1], [1,0,0], [1,1,0]], np.int32)
        y_pred = np.array([[0.2,0.6,0.7],[0.2,0.6,0.6],[0.6,0.8,0.0]], np.float32)
        metric.update_state(y_true, y_pred)
        result = metric.result()
    """
    def __init__(
        self,
        threshold = 0.5,
        name = 'heikde_skill_score',
        dtype = None
    ) -> None:

        super().__init__(
            name=name,
            dtype=dtype,
            threshold=threshold
        )

        self._true_positive = None
        self._true_negative = None
        self._positive = None
        self._predicted_positive = None
        self._negative = None
        self._predicted_negative = None
        self._sample_size = None

    def build(
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
        self._true_positive = super()._add_zeros_variable("true_positive")
        self._true_negative = super()._add_zeros_variable("true_negative")
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
            self._negative.assign(ocmc.neg())
            self._predicted_negative.assign(ocmc.pneg())
            self._positive.assign(ocmc.pos())
            self._predicted_positive.assign(ocmc.ppos())
            self._sample_size.assign(ocmc.ss())
            self._true_positive.assign(ocmc.tp())
            self._true_negative.assign(ocmc.tn())
        def manual_update() -> None:
            self._negative.assign_add(weighted_sum(1 - y_true, sample_weight))
            self._predicted_negative.assign_add(weighted_sum(1 - y_pred, sample_weight))
            self._positive.assign_add(weighted_sum(y_true, sample_weight))
            self._predicted_positive.assign_add(weighted_sum(y_pred, sample_weight))
            self._sample_size.assign_add(weighted_sum(tf.ones(tf.shape(y_true), dtype=self.dtype), sample_weight))
            self._true_positive.assign_add(weighted_sum(y_true * y_pred, sample_weight))
            self._true_negative.assign_add(weighted_sum((1 - y_true) * (1 - y_pred), sample_weight))

        tf.cond(ocmc.is_enabled(), optimized_update, manual_update)

    def result(self) -> Tensor:
        expected_correct = (self._negative * self._predicted_negative /  self._sample_size +
                self._positive * self._predicted_positive /  self._sample_size)
        return (self._true_positive + self._true_negative - expected_correct) / (self._sample_size - expected_correct)

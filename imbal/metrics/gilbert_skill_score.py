from numpy.typing import NDArray
from typing import Tuple
from tensorflow import Tensor
from imbal.metrics.util import ConfusionMatrixMetric
import keras
from keras.src.metrics import metrics_utils

@keras.saving.register_keras_serializable()
class GilbertSkillScore(ConfusionMatrixMetric):
    r"""
    Computes the Gilbert Skill Score.

    Formula:

    .. math::

       \text{gilbert_skill_score} = \frac{true\_positive - chance\_hit}{true\_positive + false\_positive +
                              false\_negative - chance\_hit}

    Note that :math:`chance\_hit` is equal to :doc:`this sub-metric </imbal/metrics/submetrics/expected_true_positive>`.

    This quotient the "skill" of a system. Assuming a system
    will not perform worse than random, the output range is :code:`[0, 1]`.
    An output of :code:`0` means the system is entirely unskilled (guessing
    randomly), while an output of :code:`1` means the system is entirely
    skilled (guessing perfectly).

    Example usage:

    .. code-block:: python

        metric = imbal.metrics.GilbertSkillScore(threshold=0.5)
        y_true = np.array([[1,1,1], [1,0,0], [1,1,0]], np.int32)
        y_pred = np.array([[0.2,0.6,0.7],[0.2,0.6,0.6],[0.6,0.8,0.0]], np.float32)
        metric.update_state(y_true, y_pred)
        result = metric.result()

    For use in TensorFlow's :code:`model.compile` function, this class
    can be passed as a class instance or as any of the following string type
    aliases:

    * :code:`"GilbertSkillScore"`
    * :code:`"gilbert_skill_score"`
    * :code:`"gs"`
    * :code:`"GS"`
    * :code:`"gss"`
    * :code:`"GSS"`

    Example:

    .. code-block:: python

       model.compile(
           optimizer="adam",
           loss="binary_crossentropy",
           metrics=["gilbert_skill_score"]
       )

    **Note:** Where appropriate, documentation for functions from :code:`tf.keras.Metric` has been
    overridden to be more descriptive. Any other non-descriptive documentation of individual functions
    on this page is due to a lack of documentation in TensorFlow's original source code. Still, TensorFlow's
    documentation and source code for the :code:`Metric` class can be found `here <https://www.tensorflow.org/api_docs/python/tf/keras/Metric>`_.

    Args:
        threshold : Optional, default :code:`0.5`. The value which a given
            prediction must be above in order to be considered a positive
            guess. All predictions below or equal to this threshold will be
            considered a negative guess.
        name : Optional, default :code:`"gilbert_skill_score"`. String name
            of the metric instance.
        dtype : Optional, default :code:`None`. Data type of the metric result.

    Returns:
        float: Gilbert skill score.
    """
    def __init__(
        self,
        threshold = 0.5,
        name = 'gilbert_skill_score',
        dtype = None
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
        self._true_negative = None
        self._direction = 'up'

    def _build(
        self,
        y_true_shape : Tuple,
        y_pred_shape : Tuple
    ) -> None:
        super()._build(y_true_shape, y_pred_shape)

        self._positive = super()._add_zeros_variable("positive")
        self._true_positive = super()._add_zeros_variable("true_positive")
        self._false_positive = super()._add_zeros_variable("false_positive")
        self._false_negative = super()._add_zeros_variable("false_negative")
        self._sample_size = super()._add_zeros_variable("sample_size")
        self._true_negative = super()._add_zeros_variable("true_negative")
        self._built = True

    def _complete_update(
            self,
            y_true: NDArray | Tensor,
            y_pred: NDArray | Tensor,
            sample_weight: NDArray | Tensor | None = None
    ):
        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self._true_positive,  # noqa: E501
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self._true_negative,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self._false_positive,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self._false_negative,  # noqa: E501
            },
            y_true,
            y_pred,
            metrics_utils.parse_init_thresholds(None, self._threshold),
            sample_weight=sample_weight
        )

        self._positive.assign(self._true_positive + self._false_negative)
        self._sample_size.assign(self._positive + self._false_positive + self._true_negative)

    def result(self) -> Tensor:
        """
        Computes the current value of the metric based on the accumulated data.

        Returns:
            The Gilbert skill score of the accumulated data.
        """
        tpr = self._positive * (self._true_positive + self._false_positive) / self._sample_size
        return (self._true_positive - tpr) / (self._true_positive + self._false_positive + self._false_negative - tpr)

from numpy.typing import NDArray
from typing import Tuple
from tensorflow import Tensor
import tensorflow as tf
from imbal.metrics.util import ConfusionMatrixMetric
from imbal.experimental.optimize_confusion_metric_callback import OptimizeConfusionMetricCallback as ocmc
from keras.src.metrics import metrics_utils

class CriticalSuccessIndex(ConfusionMatrixMetric):
    r"""
    Computes the Critical Success Index.

    Formula:

    .. math::

       \text{Critical Success Index} = \frac{true_positive}{true_positive + false_positive + false_negative}

    This quotient the "skill" of a system. The output range is :code:`[0, 1]`.
    An output of :code:`0` means the system is entirely unskilled (guessing
    randomly), while an output of :code:`1` means the system is entirely
    skilled (guessing perfectly).

    Example usage:

    .. code-block:: python

        metric = imbal.metrics.CriticalSuccessIndex(threshold=0.5)
        y_true = np.array([[1,1,1], [1,0,0], [1,1,0]], np.int32)
        y_pred = np.array([[0.2,0.6,0.7],[0.2,0.6,0.6],[0.6,0.8,0.0]], np.float32)
        metric.update_state(y_true, y_pred)
        result = metric.result()

    For use in TensorFlow's :code:`model.compile` function, this class
    can be passed as a class instance or as any of the following string type
    aliases:

    * :code:`"CriticalSuccessIndex"`
    * :code:`"critical_success_index"`
    * :code:`"csi"`
    * :code:`"CSI"`
    * :code:`"threat_score"`

    Example:

    .. code-block:: python

       model.compile(
           optimizer="adam",
           loss="binary_crossentropy",
           metrics=["CSI"]
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
        name : Optional, default :code:`"critical_success_index"`. String name
            of the metric instance.
        dtype : Optional, default :code:`None`. Data type of the metric result.

    Returns:
        float: Critical success index.
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

    def _build(
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

            metrics_utils.update_confusion_matrix_variables(
                {
                    metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self._true_positive,  # noqa: E501
                    metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self._false_positive,  # noqa: E501
                    metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self._false_negative,  # noqa: E501
                },
                y_true,
                y_pred,
                metrics_utils.parse_init_thresholds(None, self._threshold),
                sample_weight=sample_weight
            )

        tf.cond(ocmc.is_enabled(), optimized_update, manual_update)

    def result(self) -> Tensor:
        """
        Computes the current value of the metric based on the accumulated data.

        Returns:
            The Critical success index of the accumulated data.
        """
        return self._true_positive / (self._true_positive + self._false_positive + self._false_negative)

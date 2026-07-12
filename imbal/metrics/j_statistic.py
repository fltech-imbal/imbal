from imbal.metrics.true_skill_statistic import TrueSkillStatistic
import keras

@keras.saving.register_keras_serializable()
class JStatistic(TrueSkillStatistic):
    r"""
    Computes the J Statistic.

    Formula:

    .. math::

       \text{J Statistic} = true\_positive\_rate + true\_negative\_rate - 1

    Here you can find more information about :doc:`true positive rate </imbal/metrics/submetrics/true_positive_rate>`
    and :doc:`true negative rate </imbal/metrics/submetrics/true_negative_rate>`.

    This difference represents the "skill" of a system. Assuming a system
    will not perform worse than random, the output range is :code:`[0, 1]`.
    An output of :code:`0` means the system is entirely unskilled (guessing
    randomly), while an output of :code:`1` means the system is entirely
    skilled (guessing perfectly).

    Example usage:

    .. code-block:: python

        metric = imbal.metrics.JStatistic(threshold=0.5)
        y_true = np.array([[1,1,1], [1,0,0], [1,1,0]], np.int32)
        y_pred = np.array([[0.2,0.6,0.7],[0.2,0.6,0.6],[0.6,0.8,0.0]], np.float32)
        metric.update_state(y_true, y_pred)
        result = metric.result()

    Due to the following equality, the J Statistic is implemented as nothing
    more than an alias for the :doc:`True Skill Statistic </imbal/metrics/true_skill_statistic>`

    .. math::
       \begin{align}
       & \text{J Statistic} = true\_positive\_rate + true\_negative\_rate - 1 \\
       & = true\_positive\_rate - (1 - true\_negative\_rate) \\
       & = true\_positive\_rate - false\_positive\_rate \\
       & = \text{True Skill Statistic}
       \end{align}

    For use in TensorFlow's :code:`model.compile` function, this class
    can be passed as a class instance or as any of the following string type
    aliases:

    * :code:`"JStatistic"`
    * :code:`"j_statistic"`

    Example:

    .. code-block:: python

       model.compile(
           optimizer="adam",
           loss="binary_crossentropy",
           metrics=["j_statistic"]
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
        name : Optional, default :code:`"j_statistic"`. String name
            of the metric instance.
        dtype : Optional, default :code:`None`. Data type of the metric result.

    Returns:
        float: J statistic.

    """
    def __init__(
        self,
        threshold = 0.5,
        name = 'j_statistic',
        dtype = None
    ) -> None:
        super().__init__(
            name=name,
            dtype=dtype,
            threshold=threshold
        )
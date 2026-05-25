from math import floor
import keras

def optimize_metric_threshold(
    predictions,
    labels,
    metric,
    sample_weight=None,
    maximize=True,
    threshold_range=(0, 1),
    step_size=0.1
):
    """

    Finds the threshold at which a particular metric is optimized.

    Args:
        predictions: A NumPy array of predictions confidences.
        labels: A NumPy array of data labels. Should have the same shape as :code:`predictions`.
        metric: A :code:`keras.metrics.Metric` instance, or a function of the form
            :code:`f(predictions, labels, sample_weight=None)`. Note: If a Metric object with a
            pre-specified threshold is provided, the function will still work as intended. Prediction
            values are clamped to 0/1 based on the thresholds being tested before the metric value
            is calculated.
        sample_weight: Optional, default :code:`None`. Sample weights used to
            weight the metric calculation.
        maximize: Optional, default :code:`True`. Whether to maximize the metric. Should be
            set to :code:`False` if your goal is to minimize the metric.
        threshold_range: Optional, default :code:`(0, 1)`. A tuple containing the minimum and
            maximum value threshold values to test.
        step_size: Optional, default :code:`0.1`. The step size between tested threshold values.

    Returns:
        The threshold at which the provided metric is optimized.

    Example:

    .. code::

        >>> from imbal.classification import optimize_metric_threshold
        >>> import numpy as np
        >>> import keras

        >>> labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).reshape(-1, 1)
        >>> predictions = np.array([0.5, 0.2, 0.7, 0.4, 0.1, 0.3, 0.8, 0.9, 0.8, 0.6]).reshape(-1, 1)
        >>> metric = keras.metrics.F1Score(threshold=0.5)

        >>> best_threshold = optimize_metric_threshold(
        >>>     predictions,
        >>>     labels,
        >>>     metric
        >>> )

        >>> print(best_threshold)
        0.7

    """
    num_steps = int(floor(threshold_range[1] - threshold_range[0])/step_size) + 1
    thresholds = [threshold_range[0] + i*step_size for i in range(num_steps)]
    best_threshold = None
    best_metric = None
    for threshold in thresholds:
        thresholded_predictions = (predictions >= threshold).astype(int)
        if isinstance(metric, keras.Metric):
            metric.reset_state()
            metric.update_state(labels, thresholded_predictions, sample_weight=sample_weight)
            metric_value = metric.result()
        else:
            metric_value = metric(thresholded_predictions, labels, sample_weight=sample_weight)
        if (best_metric is None or
            (metric_value < best_metric and not maximize) or
            (metric_value > best_metric and maximize)):
            best_metric = metric_value
            best_threshold = threshold

    return best_threshold
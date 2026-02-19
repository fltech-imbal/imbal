from math import floor
import keras

def optimize_metric(
    predictions,
    labels,
    metric,
    sample_weight=None,
    maximize=True,
    step_size=0.1,
    threshold_range=(0, 1),
):
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
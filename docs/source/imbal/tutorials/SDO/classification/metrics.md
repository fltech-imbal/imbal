# Using `imbal.metric` In Model Classification

In model classification, metrics can be used to track model performance.
Keras includes a variety of metric object, and `imbal` provides several
additional metrics for use, a full list of which can be found on 
[this page](../../../metrics/metrics.md). This tutorial will explain some
of the ways to use metrics, along with code examples. This tutorial uses the code in the
[Balanced Fit](balanced_fit.md) tutorial as a foundation.

All the source code in this tutorial can be found at `imbal/tutorials/SDO/classification/metrics.py`.

## Common Classification Metrics

Some common classification metrics that are used are listed below, with links to their documentation.
- [keras.metrics.F1Score](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/F1Score)
- [keras.metrics.AUC](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC)
- [imbal.metrics.HeikdeSkillScore](../../../metrics/heikde_skill_score.md)
- [imbal.metrics.TrueSkillStatistic](../../../metrics/true_skill_statistic.md)
- [imbal.metrics.GilbertSkillScore](../../../metrics/gilbert_skill_score.md)
- [imbal.metrics.JStatistic](../../../metrics/j_statistic.md)
- [imbal.metrics.YoudensIndex](../../../metrics/youdens_index.md)

## How to use Metric objects

When using wither `keras` or `imbal` Metric objects in your program, there are two
main places in your code where you will use them.
1. Passing the metrics into `Model.compile`, in which case the metrics will be monitored during training
2. Using the Metric object directly, after training your model, to compute metrics on your test set.

Some considerations should be made when deciding when to use Metric objects. For example, any metric passed to
`Model.compile` will be computed every single epoch, for both the training and validation sets (if a validation
set is provided). This can be time-consuming if the computation required to compute the metric is already
inherently time-consuming (such as computing `AUC`), or if you are passing multiple metrics to `Model.compile`. For this reason, we recommend
either passing a single metric to `Model.compile`, or no metrics at all. The metric(s) passed to `Model.compile` should
only be those metrics, other than your loss function, which you are considered with monitoring during training.

## Necessary Files

- All the source code in this tutorial can be found at `imbal/tutorials/SDO/classification/metrics.py`
- The training data for this tutorial can be found at `imbal/tutorials/data/SDOBenchmark/training`
- The test data for this tutorial can be found at `imbal/tutorials/data/SDOBenchmark/test`

# Applying `Metric` Objects To Predictions

Now that we have our predictions, and have split them by their frequency, we can pass the labels and predictions
to our metric objects to gain insight into the model's performance. It is worth noting that Metric objects typically
store their results in a TensorFlow Tensor object. Therefore, we will convert the tensor to a NumPy array, and retrieve
the first value
in order to print just the metric itself, without any of the additional information about the tensor it was stored in.

```python
# Predict on test data
test_predictions = model.predict(x_test)
test_predictions = test_predictions.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Calculate metrics
f1 = metrics.F1Score(threshold=0.5)
f1.update_state(y_test, test_predictions)
print('F1 Score:', f1.result().numpy()[0])

hss = imbal.metrics.HeikdeSkillScore(threshold=0.5)
hss.update_state(y_test, test_predictions)
print('Heikde Skill Score:', hss.result().numpy()[0])

tss = imbal.metrics.TrueSkillStatistic(threshold=0.5)
tss.update_state(y_test, test_predictions)
print('True Skill Statistic:', tss.result().numpy()[0])

auroc = metrics.AUC()
auroc.update_state(y_test, test_predictions)
print('AUROC:', auroc.result().numpy())

gss = imbal.metrics.GilbertSkillScore(threshold=0.5)
gss.update_state(y_test, test_predictions)
print('Gilbert Skill Score:', gss.result().numpy()[0])

j_statistic = imbal.metrics.JStatistic(threshold=0.5)
j_statistic.update_state(y_test, test_predictions)
print('J Statistic:', j_statistic.result().numpy()[0])

youdens = imbal.metrics.YoudensIndex(threshold=0.5)
youdens.update_state(y_test, test_predictions)
print('Youden\'s Index:', youdens.result().numpy()[0])
```

The code above should yield output similar to the following:

```text
F1 Score: 0.049216997
Heikde Skill Score: 0.0042018816
True Skill Statistic: 0.065577745
AUROC: 0.6919186
Gilbert Skill Score: 0.0021053618
J Statistic: 0.065577745
Youden's Index: 0.065577745
```

Note that True Skill Statistic, J Statistic, and Youden's Index are produce the same result.
## Using `imbal.metric` In Model Regression

In model regression, metrics can be used to track model performance.
Keras includes a variety of metric object, and `imbal` provides several
additional metrics for use, a full list of which can be found on 
[this page](../../../metrics/metrics.md). This tutorial will explain some
of the ways to use metrics, along with code examples.

Throughout this tutorial, we will make use of the following metrics:
- `keras.metrics.MeanAbsoluteError`
- `keras.metrics.MeanSquaredError`
- `keras.metrics.PearsonCorrelation`

MAE, MSE, Correlation
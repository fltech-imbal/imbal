# Evaluation Metrics Tutorial (Regression)

**Full Code:** [view source code](./imbal_tutorial_metrics_regression.py)

**Train/Test Files**: [training data](./sep_model_training_regression.csv), [testing data](./sep_model_testing_regression.csv)

---

> This tutorial is based on the `balanced_fit` regression tutorial found [here](./Regular/imbal_tutorial_balanced_fit_regression_clear_sep.md)

## 1. Metrics Overview

Imbal supports various additional metrics alongside existing metrics in the `keras.metrics` library.

Additional metrics include:
1. True Skill Statistic
2. J Statistic
3. Youden's Index
4. Heikde Skill Score
5. Gilbert Skill Score
6. Critical Success Index
7. Bounded AUC

Metrics can be used in two main ways:
1. Tracking metric values at every epoch via the `metrics` parameter passed in at model compilation
2. Doing a single calculation of the metric using `model.predict(x_test)` and `metric.update_state(y_true, y_pred)`

This tutorial will explore both options.

Imbal also supports producing a true vs. predicted plot for regression style problems.

---

## 2. Model Compilation Metrics

Lightweight metrics, such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Pearson Correlation Coefficient (PCC), can be passed via the `metrics`
parameter in the model's `compile` call. These metrics can be tracked every epoch while adding minimal overhead to model training.

```python
model.compile(loss="mean_squared_error",
              optimizer="adam",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae"),
                       keras.metrics.MeanSquaredError(name="mse"),
                       keras.metrics.PearsonCorrelation(name="pcc")],
              )
```

---

## 3. Single Calculation Metrics (?)

More computationally expensive metrics, such as Bounded AUC, should be calculated once after the model has been trained, unless it is desired to track it at every epoch.

Choosing to calculate these kinds of metrics once can save training time due to less computation being needed per epoch.

```python
y_pred = model.predict(x_test)
auc_metric = imbal.metrics.BoundedAUC(num_thresholds=50)
auc_metric.update_state(y_test, y_pred)
```

---

## 4. Results

```python
results = model.evaluate(x_test, y_test)
loss, mae, mse, pcc = results

print(f"Test Loss: {loss:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test MSE: {mse:.4f}")
print(f"Test PCC: {pcc:.4f}")
```

### Example Output

![Model Results](../images/metrics_regression.png)

---

## 5. True vs. Predicted Plot

Imbal supports plotting a basic true vs. predicted values plot for regression style problems.

```python
imbal.regression.plot_true_vs_predictions(y_test, y_pred)
```

### Results

![True vs. Predicted](../images/metrics_regression_true_vs_predicted_visualizer.png)

---

## 6. AUC Curve (?)

wip
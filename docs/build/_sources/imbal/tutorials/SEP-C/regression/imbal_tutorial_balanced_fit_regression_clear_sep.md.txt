# Imbal Balanced Regression Tutorial

This tutorial demonstrates how to train a neural network for a regression task while addressing data imbalance using density-based sample weighting and the `balanced_fit` function.

### Files Needed

**Full Code:** [view source code](../../../../../../tutorials/SEP-C/regression/imbal_tutorial_balanced_fit_regression_clear_sep.py)

**Train/Test Files**: [training data](../../../../../../tutorials/data/SEP-C/sep_model_training_regression.csv), [testing data](../../../../../../tutorials/data/SEP-C/sep_model_testing_regression.csv)

---

> **Before you begin:** Use the [Tutorial Setup](imbal_tutorial_setup_regression.md) guide as your starting point, then continue with this tutorial.

## 1. Calculate Sample Densities

To address imbalance in continuous targets, we estimate label densities.

```python
labels_kde = y_train.reshape(-1).copy()
kde = imbal.regression.fit_kde(labels_kde)
densities = imbal.regression.get_sample_densities(labels_kde, kde)
```

### Explanation

* **KDE (Kernel Density Estimation)** models the distribution of target values.
* **Densities** measure how common each sample is.
* These densities are used during training to emphasize rare samples.

---

## 2. Model Compilation and Training

### Compilation

```python
model.compile(
    loss="mean_squared_error",
    optimizer="adam",
    metrics=["mae"],
)
```

### Training

```python
max_epochs = 300
batch_size = 32

model.balanced_fit(
    x_train,
    y_train,
    sample_density=densities,
    batch_size=batch_size,
    epochs=max_epochs,
)
```

### Explanation

* **Loss**: Mean Squared Error (MSE) for regression tasks.
* **Metric**: Mean Absolute Error (MAE) for interpretability.
* **balanced_fit** uses density-based weighting to better learn from imbalanced target distributions.

---

## 3. Results

### Model Evaluation

```python
results = model.evaluate(x_test, y_test)
loss, mae = results
predictions = model.predict(x_test)

print(f"Test Loss: {loss:.4f}")
print(f"Test MAE: {mae:.4f}")

threshold = np.log(10)

y_true = y_test.reshape(-1)
y_pred = predictions.reshape(-1)

common_mask = y_true < threshold
rare_mask = y_true >= threshold

common_mae = np.mean(np.abs(y_true[common_mask] - y_pred[common_mask]))
rare_mae = np.mean(np.abs(y_true[rare_mask] - y_pred[rare_mask]))

print(f"Common sample MAE (< ln(10)): {common_mae:.4f}")
print(f"Rare sample MAE (>= ln(10)): {rare_mae:.4f}")
```

### Example Output

```text
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 4.1165 - mae: 1.2846 
Test Loss: 4.1165
Test MAE: 1.2846
Common sample MAE (< ln(10)): 1.2776
Rare sample MAE (>= ln(10)): 1.6310
```

---

## 4. Visualization

The model’s predictions are visualized by plotting true values against predicted values to assess performance and highlight rare vs. frequent samples.

```python
imbal.regression.plot_true_vs_predictions(y_test,
                                          predictions,
                                          )
```

### Example Output

![Model Results](../../../../_static/tutorials/SEP-C/balanced_fit_regression_visualizer.png)

---

## 5. Optional: Using Explicit Sample Weights

Alternatively, you can manually convert the sample densities into training weights and pass those weights to `balanced_fit`. This gives you more direct control over how strongly rare samples are emphasized during training.

Replace the training call with:

```python
from imbal.regression import reciprocal_importance

weights = reciprocal_importance(densities, alpha=0.8)

model.balanced_fit(
    x_train,
    y_train,
    sample_weight=weights,
    batch_size=batch_size,
    epochs=max_epochs,
)
```

### Explanation

* `reciprocal_importance` transforms the density values into sample weights.
* Lower-density (rarer) samples receive larger weights.
* The `alpha` parameter controls how aggressively rare samples are upweighted.
* This approach is useful when you want more manual control than passing `sample_density` directly.

### Results

```text
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 2.7186 - mae: 0.9024 
Test Loss: 2.7186
Test MAE: 0.9024
Common sample MAE (< ln(10)): 0.8913
Rare sample MAE (>= ln(10)): 1.4566
```

![Model Results](../../../../_static/tutorials/SEP-C/balanced_fit_custom_alpha_regression_visualizer.png)

This optional approach gives you manual control over the weighting strategy, while the default `sample_density` workflow keeps the training setup simpler.

--- 

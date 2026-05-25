# Imbal Balanced Binary Classification Tutorial

This tutorial demonstrates how to train a neural network for a classification task while addressing data imbalance using the `balanced_fit` function.

### Files Needed

**Full Code:** [view source code](../../../../../../tutorials/SEP-C/classification/imbal_tutorial_balanced_fit_classification_clear_sep.py)

**Train/Test Files**: [training data](../../../../../../tutorials/data/SEP-C/sep_model_training_classification.csv), [testing data](../../../../../../tutorials/data/SEP-C/sep_model_testing_classification.csv)

---

> **Before you begin:** Use the [Tutorial Setup](imbal_tutorial_setup_classification.md) guide as your starting point, then continue with this tutorial.

## 1. Model Compilation and Training

### Compilation

```python
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=[
        tf.keras.metrics.F1Score(threshold=0.5, name="F1Score"),
        imbal.metrics.HeidkeSkillScore(threshold=0.5, name="HSS"),
    ],
)
```

### Training with `balanced_fit`

```python
max_epochs = 300
batch_size = 32

model.balanced_fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=max_epochs,
)
```

### Explanation

* **balanced_fit** automatically handles class imbalance internally.
* No need to manually compute sample weights.
* Training configuration remains similar to standard fitting.

---

## 2. Results

### Model Evaluation

```python
results = model.evaluate(x_test, y_test)
loss, f1_score, hss = results

print(f"Test Loss: {loss:.4f}")
print(f"Test F1Score: {f1_score:.4f}")
print(f"Test HSS: {hss:.4f}")

if model.best_metric_threshold is not None:
    best_threshold = model.best_metric_threshold
    test_predictions = model.predict(x_test)
    test_predictions = test_predictions.reshape(-1, 1)
    test_predictions = (test_predictions > best_threshold).astype(np.float32)

    best_threshold = model.best_metric_threshold
    hss = imbal.metrics.HeidkeSkillScore(threshold=best_threshold)
    hss.update_state(y_test, test_predictions)

    f1 = keras.metrics.F1Score(threshold=best_threshold)
    f1.update_state(y_test, test_predictions)

    print(
        f'Best found threshold: {model.best_metric_threshold}\n'
        f'HSS using Best Threshold: {hss.result()[0]:.4f}\n'
        f'F1Score using Best Threshold: {f1.result()[0]:.4f}\n'
    )
```

### Example Output

```text
Best decision threshold based on metric "F1Score": 0.9
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - F1Score: 0.3380 - HSS: 0.3169 - loss: 0.2065          
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 968us/step
Test Loss: 0.2065
Test F1Score: 0.3380
Test HSS: 0.3169
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 919us/step
Best found threshold: 0.9
HSS using Best Threshold: 0.5296
F1Score using Best Threshold: 0.5405
```

---

## 3. Optional: Using Class Weights

Alternatively, you can manually specify class weights during training. Replace the `model.balanced_fit` call with:

```python
class_weights = {0: 0.9, 1: 0.1}

model.balanced_fit(
    x_train,
    y_train,
    class_weight=class_weights,
    batch_size=batch_size,
    epochs=max_epochs,
)
```

### Results

```text
Best decision threshold based on metric "F1Score": 0.7
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - F1Score: 0.6857 - HSS: 0.6785 - loss: 0.0622   
Test Loss: 0.0622
Test F1Score: 0.6857
Test HSS: 0.6785
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 940us/step
Best found threshold: 0.7
HSS using Best Threshold: 0.6993
F1Score using Best Threshold: 0.7059
```

This optional approach gives you manual control over class importance, while `balanced_fit` automates the process.


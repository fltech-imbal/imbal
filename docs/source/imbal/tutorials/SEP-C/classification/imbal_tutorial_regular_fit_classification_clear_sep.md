# Imbal Binary Classification Tutorial (Regular Fit)

This tutorial walks through a complete machine learning workflow using Imbal for a binary classification task.

### Files Needed

**Full Code:** [view source code](../../../../../../tutorials/SEP-C/classification/imbal_tutorial_regular_fit_classification_clear_sep.py)

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

### Explanation

* **Loss**: Binary crossentropy is ideal for binary classification.
* **Optimizer**: Adam is efficient and widely used.
* **Metrics**:

  * **F1Score**: Balances precision and recall, useful for imbalanced datasets.
  * **HSS (Heidke Skill Score)**: Measures predictive skill relative to random chance.

### Training

```python
max_epochs = 300
batch_size = 32

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=max_epochs,
)
```

### Explanation

* **Epochs**: Number of times the model sees the entire dataset.
* **Batch size**: Number of samples processed before updating weights.

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
        f'F1Score using Best Threshold: {f1.result()[0]:.4f}\n'
        f'HSS using Best Threshold: {hss.result()[0]:.4f}\n'
    )
```

### Example Output

```text
Best decision threshold based on metric "F1Score": 0.3
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - F1Score: 0.7586 - HSS: 0.7540 - loss: 0.0554       
Test Loss: 0.0554
Test F1Score: 0.7586
Test HSS: 0.7540
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 956us/step
Best found threshold: 0.3
F1Score using Best Threshold: 0.7742
HSS using Best Threshold: 0.7695
```

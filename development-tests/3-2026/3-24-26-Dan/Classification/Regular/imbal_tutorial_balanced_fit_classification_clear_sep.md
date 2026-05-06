# Imbal Balanced Binary Classification Tutorial

This tutorial demonstrates how to train a neural network for a classification task while addressing data imbalance using the `balanced_fit` function.

### Files Needed

**Full Code:** [view source code](./imbal_tutorial_balanced_fit_classification_clear_sep.py)

**Train/Test Files**: [training data](../sep_model_training_classification.csv), [testing data](../sep_model_testing_classification.csv)

---

> **Before you begin:** Use the [Tutorial Setup](../imbal_tutorial_setup_classification.md) guide as your starting point, then continue with this tutorial.

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
```

### Example Output

![Model Results](../../images/balanced_fit_classification.png)

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

![Model Results](../../images/balanced_fit_classification_class_weights.png)

This optional approach gives you manual control over class importance, while `balanced_fit` automates the process.

--- 

# Imbal Binary Classification Tutorial (Autoencoder + Regular Fit)

This tutorial walks through a complete machine learning workflow using Imbal for a binary classification task with the **autoencoder feature** enabled.

### Files Needed

**Full Code:** [view source code](./imbal_tutorial_regular_fit_ae_classification_clear_sep.py)

**Train/Test Files**: [training data](../sep_model_training_classification.csv), [testing data](../sep_model_testing_classification.csv)

---

> **Before you begin:** Use the [Tutorial Setup](../imbal_tutorial_ae_setup_classification.md) guide as your starting point, then continue with this tutorial.

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
    generate_decoder_branch=True,
)
```

### 🔍 Autoencoder Feature

This model enables an **autoencoder branch** during training:

* `generate_decoder_branch=True` adds a decoder that reconstructs inputs.
* This encourages the network to learn **better feature representations**, improving generalization.

### Training with Regular Fit

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

* The autoencoder branch adds an auxiliary learning objective.
* This addition improves robustness and performance on imbalanced datasets.

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

![Model Results](../../images/regular_fit_ae_classification.png)

---

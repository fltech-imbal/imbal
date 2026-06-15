# Imbal Balanced Binary Classification Tutorial (Autoencoder + Balanced Fit)

This tutorial demonstrates how to train a neural network for a binary classification task while addressing class imbalance using the `balanced_fit` function with the **autoencoder feature** enabled.

### Files Needed

**Full Code:** [view source code](../../../../../../tutorials/SEP-C/classification/imbal_tutorial_balanced_fit_ae_classification_clear_sep.py)

**Train/Test Files**: [training data](../../../../../../tutorials/data/SEP-C/sep_model_training_classification.csv), [testing data](../../../../../../tutorials/data/SEP-C/sep_model_testing_classification.csv)

---

> **Before you begin:** Use the [Tutorial Setup](imbal_tutorial_ae_setup_classification.md) guide as your starting point, then continue with this tutorial.

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

### Training with Balanced Fit

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

* **balanced_fit** automatically handles class imbalance.
* The autoencoder branch adds an auxiliary learning objective.
* This combination improves robustness and performance on imbalanced datasets.

---

## 2. Results

### Model Evaluation

```python
results = model.evaluate(x_test, y_test)
loss, f1_score, hss = results

print(f"Test Loss: {loss:.4f}")
print(f"Test F1Score: {f1_score:.4f}")
print(f"Test HSS: {hss:.4f}")

if model.best_decision_threshold is not None:
    best_threshold = model.best_decision_threshold
    test_predictions = model.predict(x_test)
    test_predictions = test_predictions.reshape(-1, 1)
    test_predictions = (test_predictions > best_threshold).astype(np.float32)

    best_threshold = model.best_decision_threshold
    hss = imbal.metrics.HeidkeSkillScore(threshold=best_threshold)
    hss.update_state(y_test, test_predictions)

    f1 = keras.metrics.F1Score(threshold=best_threshold)
    f1.update_state(y_test, test_predictions)

    print(
        f'Best found threshold: {model.best_decision_threshold}\n'
        f'F1Score using Best Threshold: {f1.result()[0]:.4f}\n'
        f'HSS using Best Threshold: {hss.result()[0]:.4f}\n'
    )
```

### Example Output

```text
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - F1Score: 0.5125 - HSS: 0.4984 - loss: 0.1888  
Test Loss: 0.1888
Test F1Score: 0.5125
Test HSS: 0.4984
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 957us/step
Best found threshold: 0.9
F1Score using Best Threshold: 0.6667
HSS using Best Threshold: 0.6594
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
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - F1Score: 0.7339 - HSS: 0.7277 - loss: 0.0774  
Test Loss: 0.0774
Test F1Score: 0.7339
Test HSS: 0.7277
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 903us/step
Best found threshold: 0.9
F1Score using Best Threshold: 0.6154
HSS using Best Threshold: 0.6089
```

This optional approach gives manual control over class importance, while `balanced_fit` automates the process.

# Imbal cRT Binary Classification Tutorial

This tutorial demonstrates how to train a neural network for a classification task while addressing data imbalance using the `cRT_fit` function.

### Files Needed

**Full Code:** [view source code](../../../../../../tutorials/SEP-C/classification/imbal_tutorial_decoupled_fit_classification_clear_sep.py)

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

### Training with `cRT_fit`

```python
max_epochs = 300
batch_size = 32

model.cRT_fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=max_epochs,
)
```

### Explanation

* **cRT_fit** trains the model using the cRT-based fitting routine for imbalanced classification.
* The training call remains similar to a standard fit workflow.
* Batch size and epoch count are configured explicitly before training.

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

    if model.best_class_weights is not None:
        print(f'Best class weights: {model.best_class_weights}')

    print(
        f'Best found threshold: {model.best_decision_threshold}\n'
        f'HSS using Best Threshold: {hss.result()[0]:.4f}\n'
        f'F1Score using Best Threshold: {f1.result()[0]:.4f}\n'
    )
```

### Example Output

```text
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - F1Score: 0.4490 - HSS: 0.4336 - loss: 0.1662   
Test Loss: 0.1662
Test F1Score: 0.4490
Test HSS: 0.4336
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 784us/step
Best found threshold: 0.9
F1Score using Best Threshold: 0.6471
HSS using Best Threshold: 0.6392
```

---

## 3. Optional: Using Class Weights

Alternatively, you can manually specify class weights during training. Replace the `model.cRT_fit` call with:

```python
class_weights = {0: 0.9, 1: 0.1}

model.cRT_fit(
    x_train,
    y_train,
    class_weight=class_weights,
    batch_size=batch_size,
    epochs=max_epochs,
)
```

### Results

```text
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - F1Score: 0.6486 - HSS: 0.6403 - loss: 0.1053   
Test Loss: 0.1053
Test F1Score: 0.6486
Test HSS: 0.6403
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 722us/step
Best class weights: {0: 0.8, 1: 0.2}
Best found threshold: 0.8
F1Score using Best Threshold: 0.6667
HSS using Best Threshold: 0.6594
```

This optional approach gives you manual control over class importance, while `cRT_fit` automates the process.

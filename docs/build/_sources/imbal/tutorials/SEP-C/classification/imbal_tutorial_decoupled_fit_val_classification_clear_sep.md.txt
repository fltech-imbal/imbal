# Imbal Decoupled Binary Classification Tutorial (With a Validation Set)

This tutorial demonstrates how to train a neural network for a classification task while addressing data imbalance using the `cRT_fit` function with a validation set.

### Files Needed

**Full Code:** [view source code](../../../../../../tutorials/SEP-C/classification/imbal_tutorial_decoupled_fit_val_classification_clear_sep.py)

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
(x_train, y_train), (x_val, y_val) = imbal.classification.split(
    x_train,
    y_train,
    test_size=0.1,
)

PATIENCE = 30

model.cRT_fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val.reshape(-1, 1)),
    batch_size=batch_size,
    epochs=max_epochs,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
        )
    ],
)
```

### Explanation

* **cRT_fit** applies a decoupled training strategy to better handle class imbalance.
* A validation set is used to monitor model performance on unseen data.
* Validation data helps reduce overfitting by enabling early stopping based on validation loss.
* The best model weights are restored when validation performance stops improving.
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
        f'Best found threshold {model.best_metric_threshold}\n'
        f'F1Score using Best Threshold: {f1.result()[0]:.4f}\n'
        f'HSS using Best Threshold: {hss.result()[0]:.4f}\n'
    )
```

### Example Output

```text
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - F1Score: 0.4528 - HSS: 0.4370 - loss: 0.2092   
Test Loss: 0.2092
Test F1Score: 0.4528
Test HSS: 0.4370
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 889us/step
Best found threshold: 0.9
F1Score using Best Threshold: 0.6154
HSS using Best Threshold: 0.6059
```
---

## 3. Optional: Using Class Weights

Alternatively, you can explore different class weights during training. Replace the `model.cRT_fit` call with:

```python
class_weight_candidates = [[0.9, 0.1], [0.8, 0.2], [0.5, 0.5]]

model.cRT_fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val.reshape(-1, 1)),
    class_weight=class_weight_candidates,
    batch_size=batch_size,
    epochs=max_epochs,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
        )
    ],
)
```

### Results

```text
Restoring model weights from fit on class weight candidate at index 0
Class weights of best fit: [0.9 0.1]
Performing final fit using combined training and validation data
...  
Test Loss: 0.0544
Test F1Score: 0.6471
Test HSS: 0.6392
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step 
Best found threshold: 0.4
F1Score using Best Threshold: 0.6857
HSS using Best Threshold: 0.6785
```

> **NOTE:** at the end of training, the index of the best class weight is printed. For convenience, the associated class weight is also printed out.

This optional approach allows the function to explore different class weights, helping find the best hyperparameter values for the model while still using validation data and early stopping.

---

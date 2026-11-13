# Imbal Binary Classification with Representation Loss

This tutorial walks through a complete machine learning workflow using a representation loss in Imbal for a binary classification task.

### Files Needed

**Full Code:** [view source code](../../../../../../tutorials/SEP-C/classification/imbal_tutorial_representation_loss.py)

**Train/Test Files**: [training data](../../../../../../tutorials/data/SEP-C/sep_model_training_classification.csv), [testing data](../../../../../../tutorials/data/SEP-C/sep_model_testing_classification.csv)

---

> **Before you begin:** Use the [Tutorial Setup](imbal_tutorial_setup_classification.md) guide as your starting point, then continue with this tutorial.


## 1. Representation Loss Definition

First, we define a simple representation loss that tries to ensure that the representations of samples
of the same class are close to the mean representation of that class, while also ensuring
that the mean representations for different classes are far apart. To do this
we can use the ii-loss from [Hassen and Chan, 2020](https://cs.fit.edu/~pkc/papers/sdm20.pdf).

```python
def representation_loss_function(y_true, representations):
    y_true = tf.reshape(y_true, [-1])

    class_values, class_indices = tf.unique(y_true)
    num_classes = tf.shape(class_values)[0]

    centroids = tf.math.unsorted_segment_mean(
        representations,
        class_indices,
        num_classes
    )

    centroid_per_sample = tf.gather(centroids, class_indices)

    intra_class_loss = tf.reduce_sum(
        tf.square(
            tf.norm(representations - centroid_per_sample, axis=1)
        )
    )

    differences = centroids[:, None, :] - centroids[None, :, :]
    pairwise_distances = tf.norm(differences, axis=2)

    pairwise_distances += tf.eye(num_classes) * 1e10
    minimum_distance = tf.reduce_min(pairwise_distances)
    return intra_class_loss - minimum_distance
```

## 2. Model Compilation and Training

### Compilation

We pass the defined representation loss to `imbal.classification.Model.compile`

```python
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=[
        tf.keras.metrics.F1Score(threshold=0.5, name="F1Score"),
        imbal.metrics.HeidkeSkillScore(threshold=0.5, name="HSS"),
        representation_loss=representation_loss_function
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
Best decision threshold based on metric "F1Score": 0.1
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - F1Score: 0.6379 - HSS: 0.6295 - loss: 0.1359  
Test Loss: 0.1359
Test F1Score: 0.6379
Test HSS: 0.6295
24/24 ━━━━━━━━━━━━━━━━━━━━ 0s 929us/step
Best found threshold: 0.1
F1Score using Best Threshold: 0.4762
HSS using Best Threshold: 0.4703
```

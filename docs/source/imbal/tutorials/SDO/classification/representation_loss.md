# Classification Model Training Using Representation loss

This tutorial will explain how to use representation loss functions within `imbal`, extending
the code from the [Balanced Fit](balanced_fit.md) tutorial.

All the source code for this tutorial can be found at `imbal/tutorials/SDO/classificaiton/representation_loss.py`.

## 1. Representation Loss Definition

First, we define a simple representation loss that tries to ensure that the representations of samples
of the same class are close to the mean representation of that class, while also ensuring
that the mean representations for different classes are far apart. To do this
we can use the ii-loss from [Hassen and Chan, 2020](https://cs.fit.edu/~pkc/papers/sdm20.pdf).

```python
import tensorflow as tf
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

Compared to the original balanced fit tutorial, the only different here is
that in the call to `model.compile`, we set the representation loss
to be our previously defined loss function.

```python
"""
Compile and train model
"""
LEARNING_RATE = 5e-5
EPOCHS = 20
BATCH_SIZE = 256

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', metrics.F1Score(threshold=0.5)],
    representation_loss=representation_loss_function
)

model.balanced_fit(
    x_train,
    y_train.reshape(-1, 1),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    stratify_batches=True
)

model.evaluate(x_test, y_test.reshape(-1, 1))
```

The above code should produce the standard TensorFlow output for model
training and evaluation.

## 3. Metrics and Results Visualization

The following code outputs the overall accuracy, as well as the
accuracy for the frequent and rare data in both the training and
test sets, along with plotting confusion matrices for the
training and test set.

```python
"""
Data and results visualization
"""
KDE_BIN_COUNT=32

test_rare_mask = y_test == 1
test_frequent_mask = ~test_rare_mask
print('Number of test samples with log10 flux < -4:', np.sum(test_frequent_mask))
print('Number of test samples with log10 flux >= -4:', np.sum(test_rare_mask))

# Predict on test data
test_predictions = model.predict(x_test)
test_predictions = test_predictions.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Calculate metrics
hss = imbal.metrics.HeidkeSkillScore(threshold=0.5)
hss.update_state(y_test, test_predictions)

f1 = metrics.F1Score(threshold=0.5)
f1.update_state(y_test, test_predictions)

print(
    f'Heikde Skill Score: {hss.result()[0]:.4f}\n'
    f'F1 Score: {f1.result()[0]:.4f}\n'
)

best_threshold = model.best_decision_threshold
hss = imbal.metrics.HeidkeSkillScore(threshold=best_threshold)
hss.update_state(y_test, test_predictions)

f1 = metrics.F1Score(threshold=best_threshold)
f1.update_state(y_test, test_predictions)

print(
    f'Best threshold: {model.best_decision_threshold}\n'
    f'Heikde Skill Score using Best Threshold: {hss.result()[0]:.4f}\n'
    f'F1 Score using Best Threshold: {f1.result()[0]:.4f}\n'
)
```

Below are examples of what the generated output and plots should look 
like for the above code.

```text
Best decision threshold based on metric "f1_score": 0.5
Number of test samples with log10 flux < -4: 586
Number of test samples with log10 flux >= -4: 14
19/19 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step
Heikde Skill Score: -0.0027
F1 Score: 0.0397

Best threshold: 0.5
Heikde Skill Score using Best Threshold: -0.0027
F1 Score using Best Threshold: 0.0397
```

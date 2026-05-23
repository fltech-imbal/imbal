# Regular Classification on SDOBenchmark with Validation Data

## Necessary Files

- All the source code in this tutorial can be found at `imbal/tutorials/SDO/classification/sdo_regular_fit_val.py`
- The training data for this tutorial can be found at `imbal/tutorials/data/SDOBenchmark/training`
- The test data for this tutorial can be found at `imbal/tutorials/data/SDOBenchmark/test`

## 1. SDO Classification Setup

Before training a model on the SDOBenchmark dataset, the data must first be loaded, and
a model be initialized. The steps for doing so can be found [here](setup.md).

## 2. Create Validation Split

The code below splits the training data into a training subset and validation set, with
$90%$ of the original training data ending up in the new training subset, and $10\%$ of the
original training data ending up in the validation set. Notably, `imbal.classification.split`
performs a stratified split, aiming to maintain a similar class distribution between both
the training and validation set.

```python
"""
Create validation split
"""

(x_train, y_train), (x_val, y_val) =  imbal.regression.split(x_train, y_train, test_size=0.1)
```

## 3. Model Compilation and Training

The code below compiles the model is a manner identical to the `keras.Model`
object, then performs a model fit on the training data. We monitor when
the validation loss begins to diverge to determine when to end training,
restoring the state of the model right before the divergence began.

Notably, the
`imbal.classification.Model` object can take an extra parameter in its fit
function, called `stratify_batches`. This parameter ensures that rarer
samples are present in each batch during training.

```python
"""
Compile and train model
"""
LEARNING_RATE = 2e-4
BATCH_SIZE = 256
PATIENCE = 10

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', metrics.F1Score(threshold=0.5)],
)

history = model.fit(
    x_train,
    y_train.reshape(-1, 1),
    # validation_data=(x_val, y_val.reshape(-1, 1)),
    validation_split=0.1,
    epochs=500,
    batch_size=BATCH_SIZE,
    stratify_batches=True, # Ensure all batches have a similar data distribution
    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)]
)

print(f'Fit stopped after {len(history.history["loss"])} epochs')
print(f'Restored weights from epoch {len(history.history["loss"]) - PATIENCE}')

model.evaluate(x_test, y_test.reshape(-1, 1))
```

The above code should produce the standard TensorFlow output for model
training and evaluation, followed by something similar to:

```text
Fit stopped after 81 epochs
Restored weights from epoch 71
```

## 4. Metrics and Results Visualization

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
hss = imbal.metrics.HeikdeSkillScore(threshold=0.5)
hss.update_state(y_test, test_predictions)

f1 = metrics.F1Score(threshold=0.5)
f1.update_state(y_test, test_predictions)

print(
    f'Heikde Skill Score: {hss.result()[0]:.4f}\n'
    f'F1 Score: {f1.result()[0]:.4f}\n'
)

imbal.classification.plot_confusion_matrix(
    y_test,
    test_predictions,
    save_figure='sample-sdo-regular-fit-val-confusion-matrix.png'
)

imbal.classification.plot_roc(
    y_test,
    test_predictions,
    save_figure='sample-sdo-regular-fit-val-roc.png'
)
```

Below are examples of what the generated output and plots should look 
like for the above code.

```text
Number of test samples with log10 flux < -4: 586
Number of test samples with log10 flux >= -4: 14
19/19 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
Heikde Skill Score: 0.0000
F1 Score: 0.0000
```

<div style="display: flex; gap: 8px; max-width: 100%;">
<img style="flex:1; max-width: 49%;" src="../../../../_static/tutorials/SDO/sample-sdo-regular-fit-val-confusion-matrix.png"/>
<img style="flex:1; max-width: 49%;" src="../../../../_static/tutorials/SDO/sample-sdo-regular-fit-val-roc.png"/>
</div>

### Optional: Validation via `validation_split`

By commenting out the code in section two, and modifying the commented code in section three:

```python
# In section 2...

# (x_train, y_train), (x_val, y_val) =  imbal.classification.split(x_train, y_train, test_size=0.1)

# ... the during fit (section 3) ...

history = model.fit(
    x_train,
    y_train.reshape(-1, 1),
    # validation_data=(x_val, y_val.reshape(-1, 1)),
    validation_split=0.1,
    epochs=500,
    batch_size=BATCH_SIZE,
    stratify_batches=True, # Ensure all batches have a similar data distribution
    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)]
)
```

we get the following results:

```text
Best decision threshold based on metric "f1_score": 0.1
Fit stopped after 14 epochs
Restored weights from epoch 4
19/19 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.9767 - f1_score: 0.0000e+00 - loss: 0.1446
Number of test samples with log10 flux < -4: 586
Number of test samples with log10 flux >= -4: 14
19/19 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step
Heikde Skill Score: 0.0000
F1 Score: 0.0000

Best threshold: 0.1
Heikde Skill Score using Best Threshold: 0.0000
F1 Score using Best Threshold: 0.0000
```

<div style="display: flex; gap: 8px; max-width: 100%;">
<img style="flex:1; max-width: 49%;" src="../../../../_static/tutorials/SDO/sample-sdo-regular-fit-val-confusion-matrix-split.png"/>
<img style="flex:1; max-width: 49%;" src="../../../../_static/tutorials/SDO/sample-sdo-regular-fit-val-roc-split.png"/>
</div>
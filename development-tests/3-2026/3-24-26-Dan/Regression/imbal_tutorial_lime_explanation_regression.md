# LIME Explanation Tutorial (Regression)

**Full Code:** [view source code](imbal_tutorial_lime_explanation_regression.py)

**Train/Test Files**: [training data](./sep_model_training_regression.csv), [testing data](./sep_model_testing_regression.csv)

---

> This core code is a sample from the `balanced_fit` tutorial code found [here](./Regular/imbal_tutorial_balanced_fit_regression_clear_sep.md)

## 1. Core Code

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import layers

import imbal

seed = 42
tf.keras.utils.set_random_seed(seed)

target_column = "ln_peak_intensity"

max_epochs = 300
batch_size = 32

train_data = pd.read_csv("sep_model_training_regression.csv")
test_data = pd.read_csv("sep_model_testing_regression.csv")

y_train = train_data[target_column].values.reshape(-1, 1).astype("float32")
y_test = test_data[target_column].values.reshape(-1, 1).astype("float32")

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test = test_data.drop(columns=[target_column]).values.astype(np.float32)

def build_model(input_shape: int) -> imbal.regression.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu")(inputs)
    hidden2 = layers.Dense(12, activation="relu")(hidden1)
    hidden3 = layers.Dense(8, activation="relu")(hidden2)
    hidden4 = layers.Dense(6, activation="relu")(hidden3)
    outputs = layers.Dense(1)(hidden4)

    return imbal.regression.Model(inputs=inputs, outputs=outputs, name="sep_model")

model = build_model(x_train.shape[1])

labels_kde = y_train.reshape(-1).copy()
kde = imbal.regression.fit_kde(labels_kde)
densities = imbal.regression.get_sample_densities(labels_kde, kde)

model.compile(
    loss="mean_squared_error",
    optimizer="adam",
    metrics=["mae"],
)

from imbal.regression import reciprocal_importance
weights = reciprocal_importance(densities, alpha=0.8)

model.balanced_fit(
    x_train,
    y_train,
    sample_weight=weights,
    batch_size=batch_size,
    epochs=max_epochs,
)
```

---

## 2. LIME Explanations

### A. Correct Prediction Example

```python
labels = train_data.drop(columns=[target_column]).columns.tolist()

target_sample_index = -1

imbal.regression.lime_explain_tabular_sample(
    x_test[target_sample_index],
    model,
    x_train,
    actual_label=np.round(float(y_test.reshape(-1)[target_sample_index]), 4),
    feature_names=labels,
)
```

### Explanation

This example explains a **correct regression prediction**.

* `labels` stores the feature names for readability.
* `target_sample_index = -1` selects a sample near the end of the dataset.
* `lime_explain_tabular_sample(...)` generates a local explanation for that prediction.

Instead of class probabilities, we interpret **how features push the prediction higher or lower**.

---

### B. Incorrect Prediction Example

```python
target_sample_index = -5

imbal.regression.lime_explain_tabular_sample(
    x_test[target_sample_index],
    model,
    x_train,
    actual_label=np.round(float(y_test.reshape(-1)[target_sample_index]), 4),
    feature_names=labels,
)
```

### Explanation

This example shows a **numerically incorrect prediction** that also leads to a **functional classification error**.

An event is defined as:

**Event if:**

ln_peak_intensity ≥ ln(10)

In this case:

* The **true value exceeds the threshold (event)**
* The **predicted value is below the threshold (no event)**

So even though this is regression, it results in a **missed detection**.

---

## 3. Results

### Correct Prediction

![Correct Prediction](../images/lime_regression.png)

### Incorrect Prediction

![Incorrect Prediction](../images/lime_regression_wrong_prediction.png)

---

## 4. Understanding the LIME Output (Regression)

The LIME visualization explains **why the model predicted the value that it did**.

### Layout Guide (how to read the figure)

* **Top left** shows the **predicted value** (continuous output).
* **Top right** shows the **feature contribution values** (weights).

  * **Orange (positive)** → pushes the prediction **higher**
  * **Blue (negative)** → pushes the prediction **lower**
* **Bottom middle** shows the **feature values** for this specific sample.

Note: The **feature values** in this model are not the actual values (e.g. speed) but rather normalized values between 0 and 1 or -1 and 1.

---

## 5. Interpreting the Difference Between the Two Explanations

### Correct Prediction: Why the model predicted a high value successfully

In the first explanation, the model predicts a **high value** that correctly exceeds the event threshold.

* **Predicted: ~5.61**
* **Actual: ~8.69**

Even though the prediction is slightly lower than the true value, it is still **well above ln(10)**, so the event is correctly detected.

The strongest feature contributions are concentrated on the **positive (upward)** side, especially:

* `CME_DONKI_speed_norm`
* `Halo`
* `CME_CDAW_LinearSpeed_norm`

These features all push the prediction higher, and their values are relatively strong for this sample:

* `CME_DONKI_speed_norm = 0.78`
* `Halo = 1.00`
* `CME_CDAW_LinearSpeed_norm = 0.85`

This makes the prediction locally consistent: the most influential features reinforce each other, giving the model a clear signal for a **high-intensity event**.

---

### Incorrect Prediction: Why the model predicted too low

In the second explanation, the model predicts a value that falls **below the event threshold**, even though the actual value corresponds to an event.

* **Predicted: ~2.24**
* **Actual: ~4.76**

Since **2.24 < ln(10)**, the model fails to detect the event.

The important difference is that the strongest local evidence is now **less consistently pushing upward**. Instead, the explanation shows a more mixed pattern.

#### Key differences in the feature contributions

**Weaker positive drivers:**

* `CME_DONKI_speed_norm = 0.34` (much lower than 0.78 in the correct case)

→ One of the strongest upward-driving signals is significantly reduced.

**Stronger negative influence:**

* `CME_DONKI_longitude_norm` contributes negatively (blue)

→ This actively pulls the prediction downward.

**Mixed feature signals:**

Some features still push upward:

* `Halo`
* `CME_CDAW_LinearSpeed_norm`

But others push downward:

* `CME_DONKI_longitude_norm`
* additional threshold-based splits

→ The explanation is **less aligned** compared to the correct case.

**Shift in feature importance:**

* `DONKI_half_width_norm` becomes a stronger contributor
* However, its effect does not compensate for the weakened speed signal

---

### Why the prediction might have been wrong

The likely reason for the error is that the **local feature pattern is conflicting**.

In the correct example, the top contributors mostly agree and strongly push the prediction upward. In the incorrect example, the contributions are split between increasing and decreasing the prediction, and the **downward forces are strong enough to dominate**.

A few key takeaways:

* The model relies heavily on `CME_DONKI_speed_norm` as a primary signal.
* When that signal is weak, other features can override the prediction.
* Negative contributions (like longitude effects) can significantly suppress the output.
* Even strong positive signals (like `Halo = 1.00`) may not be enough on their own.

This suggests the model interprets this sample as more similar to **lower-intensity events** in the training data, leading to a missed detection despite the true label being above the threshold.

---

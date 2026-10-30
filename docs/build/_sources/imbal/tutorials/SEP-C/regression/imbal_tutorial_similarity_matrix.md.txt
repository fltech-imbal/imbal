# Similarity Matrix Visualization Tutorial (Regression)

**Full Code:** [view source code](../../../../../../tutorials/SEP-C/regression/imbal_tutorial_similarity_matrix_regression.py)

**Train/Test Files**: [training data](../../../../../../tutorials/data/SEP-C/sep_model_training_regression.csv), [testing data](../../../../../../tutorials/data/SEP-C/sep_model_testing_regression.csv)

---

## 1. Core Code

> This core code is a sample from the `balanced_fit` tutorial code found [here](imbal_tutorial_balanced_fit_regression_clear_sep.md)

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import layers

import imbal

seed = 42
tf.keras.utils.set_random_seed(
  seed
)

target_column = "ln_peak_intensity"

max_epochs = 300
batch_size = 32

train_data = pd.read_csv("../../../../../../tutorials/SEP-C/regression/sep_model_training_regression.csv")
test_data = pd.read_csv("../../../../../../tutorials/SEP-C/regression/sep_model_testing_regression.csv")

y_train = train_data[target_column].values.reshape(-1, 1).astype("float32")
y_test = test_data[target_column].values.reshape(-1, 1).astype("float32")

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test = test_data.drop(columns=[target_column]).values.astype(np.float32)


def build_model(input_shape: int) -> imbal.regression.Model:
  inputs = keras.Input(shape=(input_shape,), name="features")
  hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
  hidden2 = layers.Dense(12, activation="relu", name="hidden_layer2")(hidden1)
  hidden3 = layers.Dense(8, activation="relu", name="hidden_layer3")(hidden2)
  hidden4 = layers.Dense(6, activation="relu", name="hidden_layer4")(hidden3)
  outputs = layers.Dense(1, name="output_layer")(hidden4)
  built_model = imbal.regression.Model(inputs=inputs, outputs=outputs, name="sep_model")
  return built_model


model = build_model(x_train.shape[1])

labels_kde = y_train.reshape(-1).copy()
kde = imbal.regression.fit_kde(labels_kde)
densities = imbal.regression.get_sample_densities(labels_kde, kde)

model.compile(loss="mean_squared_error",
              optimizer="adam",
              metrics=["mae"],
              )

from imbal.regression import reciprocal_importance

weights = reciprocal_importance(densities, alpha=0.8)
model.balanced_fit(x_train,
                   y_train,
                   sample_weight=weights,
                   batch_size=batch_size,
                   epochs=max_epochs,
                   )
```

---

## 2. Similarity Matrix Visualization

```python
imbal.regression.plot_similarity_matrix(
    model,
    x_train,
    y_train,
    bins=32
)

imbal.regression.plot_similarity_matrix(
    model,
    x_test,
    y_test,
    bins=32
)
```

### Explanation

The `plot_similarity_matrix` function generates a plot that visualizes the similarity/distance
between samples in the representation space. The x and y axes of the plot indicate
the labels of samples in each bin (square). The colors on the plot indicate the similarity/distance
between samples of the label indicated by the x-axis with samples of the label indicated
by the y-axis. Ideally, samples close to the positive diagonal (as in, of the same label) have
high similarity (low distance), while those farther away from the positive diagonal have
low similarity (high distance).

---

## 3. Results

### Training Data

![Model Results](../../../../_static/tutorials/SEP-C/similarity_matrix_regression_train.png)

### Testing Data

![Model Results](../../../../_static/tutorials/SEP-C/similarity_matrix_regression_test.png)



---
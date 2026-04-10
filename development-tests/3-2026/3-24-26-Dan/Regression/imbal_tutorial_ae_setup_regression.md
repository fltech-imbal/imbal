# Tutorial Setup

### Initial Steps for Imbal Regression Tutorials w/ AE

This code is the base for each of the regression tutorials that use the autoencoder feature. Start here before continuing with any tutorial.

---

## 1. Import Packages

We begin by importing the required libraries for numerical computation, data handling, and model building.

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import layers

import imbal
```

### Explanation

* **NumPy**: Efficient numerical operations.
* **Pandas**: Loading and manipulating tabular data.
* **TensorFlow / Keras**: Deep learning framework.
* **layers**: Building blocks for neural networks.
* **imbal**: Custom regression model wrapper.

Set a random seed for reproducibility:

```python
seed = 42
tf.keras.utils.set_random_seed(seed)
```

---

## 2. Load Data

Load training and testing datasets from CSV files.

```python
target_column = "ln_peak_intensity"

train_data = pd.read_csv("sep_model_training_regression.csv")
test_data  = pd.read_csv("sep_model_testing_regression.csv")
```

### Prepare Features and Labels

```python
y_train = train_data[target_column].values.reshape(-1, 1).astype("float32")
y_test  = test_data[target_column].values.reshape(-1, 1).astype("float32")

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test  = test_data.drop(columns=[target_column]).values.astype(np.float32)
```

### Explanation

* Labels are reshaped to **(n, 1)** for compatibility with the model.
* Features and labels are converted to `float32` for TensorFlow efficiency.

---

## 3. Build the Model

We define a neural network suitable for regression.

```python
def build_model(input_shape: int) -> imbal.regression.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
    hidden2 = layers.Dense(12, activation="relu", name="hidden_layer2")(hidden1)
    hidden3 = layers.Dense(8, activation="relu", name="hidden_layer3")(hidden2)
    hidden4 = layers.Dense(6, activation="relu", name="hidden_layer4")(hidden3)
    flatten = layers.Flatten()(hidden4)
    outputs = layers.Dense(1, name="output_layer")(flatten)
    
    built_model = imbal.regression.Model(
        inputs=inputs, 
        outputs=outputs, 
        name="sep_model"
    )
    return built_model

model = build_model(x_train.shape[1])
```

### Explanation

* Input layer matches the number of features.
* Hidden layers progressively reduce dimensionality.
* ReLU activations introduce non-linearity.
* Output layer is linear (no activation), which is appropriate for regression.

---

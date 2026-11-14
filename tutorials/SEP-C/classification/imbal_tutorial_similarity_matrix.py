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

# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv("../../data/SEP-C/sep_model_training_regression.csv")
test_data  = pd.read_csv("../../data/SEP-C/sep_model_testing_regression.csv")

y_train = train_data[target_column].values.astype("float32")
y_train = np.eye(3)[np.digitize(y_train, [-1.609, np.log(10)])]
y_test  = test_data[target_column].values.astype("float32")
y_test = np.eye(3)[np.digitize(y_test, [-1.609, np.log(10)])]

print(y_test)


x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test  = test_data.drop(columns=[target_column]).values.astype(np.float32)

# ----------------------------
# Model
# ----------------------------
def build_model(input_shape: int) -> imbal.classification.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
    hidden2 = layers.Dense(12, activation="relu", name="hidden_layer2")(hidden1)
    hidden3 = layers.Dense(8, activation="relu", name="hidden_layer3")(hidden2)
    hidden4 = layers.Dense(6, activation="relu", name="hidden_layer4")(hidden3)
    flatten = layers.Flatten()(hidden4)
    outputs = layers.Dense(3, activation="softmax", name="output_layer")(flatten)
    built_model = imbal.classification.Model(inputs=inputs, outputs=outputs, name="sep_model")
    return built_model

model = build_model(x_train.shape[1])

# ----------------------------
# Training
# ----------------------------
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam"
)

model.balanced_fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=max_epochs,
)

# ----------------------------
# Visualization
# ----------------------------

imbal.classification.plot_similarity_matrix(
    model,
    x_train,
    y_train
)

imbal.classification.plot_similarity_matrix(
    model,
    x_test,
    y_test
)
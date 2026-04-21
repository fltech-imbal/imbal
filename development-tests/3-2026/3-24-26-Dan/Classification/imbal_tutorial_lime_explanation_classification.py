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
train_data = pd.read_csv("sep_model_training_classification.csv")
test_data  = pd.read_csv("sep_model_testing_classification.csv")

y_train = train_data[target_column].values.reshape(-1, 1).astype("float32")
y_test  = test_data[target_column].values.reshape(-1, 1).astype("float32")

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
    outputs = layers.Dense(1, activation="sigmoid", name="output_layer")(flatten)
    built_model = imbal.classification.Model(inputs=inputs, outputs=outputs, name="sep_model")
    return built_model

model = build_model(x_train.shape[1])


# ----------------------------
# Training
# ----------------------------
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              generate_decoder_branch=True,
              )

class_weights = {0: 0.9, 1: 0.1}

model.balanced_fit(x_train,
          y_train,
          class_weight=class_weights,
          batch_size=batch_size,
          epochs=max_epochs,
          )


# ----------------------------
# Visualization
# ----------------------------

labels = train_data.drop(columns=[target_column]).columns.tolist()

target_sample_index = -1 # Get a rare sample, which is at the end of the data

imbal.classification.lime_explain_tabular_sample(
    x_test[target_sample_index],
    model,
    x_train,
    actual_label=int(y_test.reshape(-1)[target_sample_index]),
    class_names=['Common', 'Rare'],
    feature_names=labels,
    figure_save_path="lime-explanation.html"
)

target_sample_index = -9 # Get an incorrectly predicted rare sample

imbal.classification.lime_explain_tabular_sample(
    x_test[target_sample_index],
    model,
    x_train,
    actual_label=int(y_test.reshape(-1)[target_sample_index]),
    class_names=['Common', 'Rare'],
    feature_names=labels,
    figure_save_path="lime-explanation-wrong.html"
)

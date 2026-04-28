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

# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv("sep_model_training_regression.csv")
test_data = pd.read_csv("sep_model_testing_regression.csv")

y_train = train_data[target_column].values.reshape(-1, 1).astype("float32")
y_test = test_data[target_column].values.reshape(-1, 1).astype("float32")

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test = test_data.drop(columns=[target_column]).values.astype(np.float32)

# ----------------------------
# Model
# ----------------------------
def build_model(input_shape: int) -> imbal.regression.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
    hidden2 = layers.Dense(12, activation="relu", name="hidden_layer2")(hidden1)
    hidden3 = layers.Dense(8, activation="relu", name="hidden_layer3")(hidden2)
    hidden4 = layers.Dense(6, activation="relu", name="hidden_layer4")(hidden3)
    outputs = layers.Dense(1, name="output_layer")(hidden4)
    built_model = imbal.regression.Model(
        inputs=inputs,
        outputs=outputs,
        name="sep_model",
    )
    return built_model

model = build_model(x_train.shape[1])

# ----------------------------
# Training
# ----------------------------
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

# ----------------------------
# Visualization
# ----------------------------
labels = train_data.drop(columns=[target_column]).columns.tolist()

target_sample_index = -1  # choose a rare sample to explain

imbal.regression.shap_explain_tabular_sample(
    x_test[target_sample_index],
    model,
    x_train,
    actual_label=np.round(float(y_test.reshape(-1)[target_sample_index]), 4),
    feature_names=labels,
    figure_save_path="shap-explanation.png",
    plot_type="waterfall",
)

# Sample with incorrect prediction
target_sample_index = -5  # choose an incorrect prediction of a rare sample to explain

imbal.regression.shap_explain_tabular_sample(
    x_test[target_sample_index],
    model,
    x_train,
    actual_label=np.round(float(y_test.reshape(-1)[target_sample_index]), 4),
    feature_names=labels,
    figure_save_path="shap-explanation-wrong.png",
    plot_type="waterfall",
)

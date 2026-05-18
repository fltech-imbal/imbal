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

max_epochs = 500
batch_size = 32

# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv("../../data/SEP-C/sep_model_training_classification.csv")
test_data  = pd.read_csv("../../data/SEP-C/sep_model_testing_classification.csv")

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
    outputs = layers.Dense(1, activation="sigmoid", name="output_layer")(hidden4)
    built_model = imbal.classification.Model(inputs=inputs, outputs=outputs, name="sep_model")
    return built_model

model = build_model(x_train.shape[1])

# ----------------------------
# Validation Set
# ----------------------------
(x_train, y_train), (x_val, y_val) =  imbal.classification.split(x_train, y_train, test_size=0.1)

# ----------------------------
# Training
# ----------------------------
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=[tf.keras.metrics.F1Score(threshold=0.5, name="F1Score"),
                       imbal.metrics.HeidkeSkillScore(threshold=0.5, name="HSS")],
              )

PATIENCE = 30

model.cRT_fit(x_train,
              y_train,
              validation_data=(x_val, y_val.reshape(-1, 1)),
              batch_size=batch_size,
              epochs=max_epochs,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)]
              )

# OPTIONAL: Use custom class weights during training
# Dictionary mapping classes to weights. In this case, 9:1 ratio of common:rare samples,
# making rare samples more important to the model loss function than with standard sampling.
# In this case, rare samples will contribute 10% of the loss per epoch, while common samples contribute 90%.
# NOTE: Comment above call before running the below call.

# weight pairs represent [common_class_weight, rare_class_weight]
class_weight_candidates = [[0.9, 0.1,], [0.8, 0.2], [0.5, 0.5]]

# model.cRT_fit(x_train,
#               y_train,
#               validation_data=(x_val, y_val.reshape(-1, 1)),
#               class_weight=class_weight_candidates,
#               batch_size=batch_size,
#               epochs=max_epochs,
#               callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)]
#               )


# ----------------------------
# Evaluation
# ----------------------------
results = model.evaluate(x_test, y_test)
loss, f1_score, hss = results

print(f"Test Loss: {loss:.4f}")
print(f"Test F1Score: {f1_score:.4f}")
print(f"Test HSS: {hss:.4f}")

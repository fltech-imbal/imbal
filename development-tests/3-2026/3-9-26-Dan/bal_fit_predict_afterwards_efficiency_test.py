import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import time
import tensorflow as tf
import imbal
from custom_callbacks import ConvergenceStopping

seed = 67
tf.keras.utils.set_random_seed(
    seed
)

target_column = "ln_peak_intensity"
threshold = np.log(10.0)

train_data = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_training.csv")
test_data = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_testing.csv")

y_train = (train_data[target_column].values >= threshold)
y_test = (test_data[target_column].values >= threshold)

y_train = y_train.reshape(-1, 1).astype("float32")
y_test = y_test.reshape(-1, 1).astype("float32")

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test = test_data.drop(columns=[target_column]).values.astype(np.float32)

sample_weights = imbal.classification.generate_sample_weights(y_train)

def build_model(input_shape: int) -> imbal.classification.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
    hidden2 = layers.Dense(12, activation="relu", name="hidden_layer2")(hidden1)
    hidden3 = layers.Dense(8, activation="relu", name="hidden_layer3")(hidden2)
    hidden4 = layers.Dense(6, activation="relu", name="hidden_layer4")(hidden3)
    outputs = layers.Dense(1, activation="sigmoid", name="output_layer")(hidden4)
    built_model = imbal.classification.Model(inputs=inputs, outputs=outputs, name="one_hidden_layer_6_units")
    return built_model

import random
import os

def reset_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

reset_seeds(seed)

convergence_stop = ConvergenceStopping("loss", 0.001, patience=30, restore_best_weights=True, best_weight_identifier="val_loss")

model = build_model(x_train.shape[1])

#f1 = tf.keras.metrics.F1Score(threshold=0.9)
#auroc = tf.keras.metrics.AUC(curve="ROC", name="auroc")
#fp = tf.keras.metrics.FalsePositives(thresholds=0.5, name="fp")
#fn = tf.keras.metrics.FalseNegatives(thresholds=0.5, name="fn")

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              )

start_cpu = time.process_time()

train_split, val_split = imbal.classification.split(x_train, y_train, sample_weights=sample_weights, test_size=0.2, seed=seed, shuffle=True)

x_train, y_train, sample_weights_train = train_split
x_val, y_val, sample_weights_val = val_split

model.balanced_fit(
    x_train,
    y_train,
    sample_weight=sample_weights_train,
    epochs=10000,
    batch_size=512,
    validation_data=(x_val, y_val, sample_weights_val),
    stratify_batches=True,
    callbacks=[convergence_stop],
)

# predictions
y_pred_prob = model.predict(x_test, batch_size=512)
y_pred = (y_pred_prob >= 0.5).astype(int)

# metrics
f1 = tf.keras.metrics.F1Score(threshold=0.5)
auroc = tf.keras.metrics.AUC(curve="ROC")
fp = tf.keras.metrics.FalsePositives(thresholds=0.5)
fn = tf.keras.metrics.FalseNegatives(thresholds=0.5)

# update metrics
f1.update_state(y_test, y_pred_prob)
auroc.update_state(y_test, y_pred_prob)
fp.update_state(y_test, y_pred_prob)
fn.update_state(y_test, y_pred_prob)

print("F1:", f1.result().numpy())
print("AUROC:", auroc.result().numpy())
print("False Positives:", fp.result().numpy())
print("False Negatives:", fn.result().numpy())

end_cpu = time.process_time()

cpu_time_seconds = end_cpu - start_cpu
print(f"CPU time spent: {cpu_time_seconds:.4f} seconds")

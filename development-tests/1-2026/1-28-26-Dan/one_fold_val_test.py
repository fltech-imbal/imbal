
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import layers

import imbal
from one_fold_validation import LocalMinimumStopping
from one_fold_validation import find_ideal_epoch_one_fold
from one_fold_validation import make_one_fold_split

seed = 42
tf.keras.utils.set_random_seed(
    seed
)

target_column = "ln_peak_intensity"
threshold = np.log(10.0)

num_folds_for_split = 5
max_epochs = 1000
batch_size = 512
seed = 42


def build_model(input_shape: int) -> keras.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
    hidden2 = layers.Dense(12, activation="relu", name="hidden_layer2")(hidden1)
    hidden3 = layers.Dense(8,  activation="relu", name="hidden_layer3")(hidden2)
    hidden4 = layers.Dense(6,  activation="relu", name="hidden_layer4")(hidden3)
    outputs = layers.Dense(1, activation="sigmoid", name="output_layer")(hidden4)
    return keras.Model(inputs=inputs, outputs=outputs, name="sep_model")


def make_compile_params():
    f1 = tf.keras.metrics.F1Score(threshold=0.5)
    auroc = tf.keras.metrics.AUC(curve="ROC", name="auroc")

    return imbal.classification.wrap_model_compile_parameters(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=[f1, auroc]
    )


# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_training.csv")
test_data  = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_testing.csv")

y_train = (train_data[target_column].values >= threshold).astype(int).reshape(-1, 1).astype("float32")
y_test  = (test_data[target_column].values  >= threshold).astype(int).reshape(-1, 1).astype("float32")

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test  = test_data.drop(columns=[target_column]).values.astype(np.float32)

model = build_model(x_train.shape[1])

compile_parameters = make_compile_params()

# ----------------------------
# Get split so you can make validation data
# ----------------------------

train, val, _ = make_one_fold_split(x_train, y_train, num_folds_for_split=num_folds_for_split, random_seed=seed)
x_tr, y_tr = train
x_va, y_va = val

sample_weights_val = imbal.classification.generate_sample_weights(y_tr)
sample_weights = imbal.classification.generate_sample_weights(y_train)

# ----------------------------
# Fitting
# ----------------------------
find_ideal_epoch_one_fold(
    model=model,
    x_train=x_train,
    y_train=y_train,
    x_tr_split=x_tr,
    y_tr_split=y_tr,
    x_val_split=x_va,
    y_val_split=y_va,
    callbacks=[LocalMinimumStopping(delta=0.001, patience=30)],
    fit_fn=lambda m, **kw: imbal.classification.balanced_fit(m, **kw),
    fit_kwargs={
        "sample_weights": sample_weights,
        "compile_parameters": compile_parameters,
    },
)


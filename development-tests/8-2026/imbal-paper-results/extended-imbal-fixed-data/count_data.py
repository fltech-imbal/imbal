"""
Import packages
"""
import imbal
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import pandas as pd
from tools import FitType
"""
Set script parameters
"""

LEARNING_RATE = 5e-5
FIT = FitType.DECOUPLED
VALIDATION_DATA = True
AE = True
AE_THIRD_TO_LAST = True
WEIGHT_CANDIDATES = True

REPRESENTATION_LAYER_INDEX = -2
EARLY_STOPPING_PATIENCE = 20
EPOCHS = 1000 if VALIDATION_DATA else 200

DATA_PATH = 'sep_e_log_normalized'

# Will be mostly left unchanged
STRATIFY = True
BATCH_SIZE = 2048
KDE_BIN_COUNT=64
SEED = 42

"""
Load data
"""

def load_sep_ec(path_prefix):
    training_data = pd.read_csv(path_prefix + '_training.csv')
    test_data = pd.read_csv(path_prefix + '_test.csv')
    val_data = pd.read_csv(path_prefix + '_validation.csv')
    training_labels = training_data.pop("delta_log_Intensity")
    val_labels = val_data.pop("delta_log_Intensity")
    test_labels = test_data.pop("delta_log_Intensity")
    training_data = training_data.to_numpy()
    val_data = val_data.to_numpy()
    test_data = test_data.to_numpy()
    training_labels = training_labels.to_numpy()
    val_labels = val_labels.to_numpy()
    test_labels = test_labels.to_numpy()
    return (training_data, training_labels), (val_data, val_labels), (test_data, test_labels)

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_sep_ec(
    f"SEP-E/{DATA_PATH}",
)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

common_mask = (y_train >= -0.5) & (y_train <= 0.5)
print(y_train[common_mask].shape)
print(y_train[~common_mask].shape)
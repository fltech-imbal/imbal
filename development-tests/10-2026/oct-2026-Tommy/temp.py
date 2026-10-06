from tools import FitType, load_sep_c_data, generate_plots, build_sep_ec_model, generate_weights, pcc, plot_similarity
import tensorflow as tf
import pandas as pd
import imbal

DATA_PATH = 'cleaned-dtw-SEP-EC-data'
DATA_PREFIX = 'sep_e_log_normalized'

"""
Load data
"""


def load_sep_ec(path_prefix):
    training_data = pd.read_csv(path_prefix + '_training.csv')
    test_data = pd.read_csv(path_prefix + '_test.csv')
    val_data = pd.read_csv(path_prefix + '_validation.csv')

    training_labels = training_data.pop("p16.4_tplus6")
    val_labels = val_data.pop("p16.4_tplus6")
    test_labels = test_data.pop("p16.4_tplus6")
    training_data = training_data.to_numpy()
    val_data = val_data.to_numpy()
    test_data = test_data.to_numpy()
    training_labels = training_labels.to_numpy()
    val_labels = val_labels.to_numpy()
    test_labels = test_labels.to_numpy()
    return (training_data, training_labels), (val_data, val_labels), (test_data, test_labels)


(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_sep_ec(
    f"{DATA_PATH}/{DATA_PREFIX}"
)

_, subset = imbal.regression.split(x_test, y_test, test_size=.4)
x_test, y_test = subset

print(y_test.shape)

model = tf.keras.models.load_model(f"models-final/sep_e_log_normalized_w_validation_cauchy_schwartz_global_1.keras")

plot_similarity(
    model,
    x_test,
    y_test
)

# DATA_PATH = 'cleaned-dtw-SEP-EC-data'
# DATA_PREFIX = 'sep_e_log_normalized'
# USE_DELTA = False
# VALUE_MIN = -10
# VALUE_MAX = 5.5
# VAL_DELTA = 1e-6
#
# """
# Load data
# """
#
# (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_sep_ec_data(
#     f"{DATA_PATH}/{DATA_PREFIX}"
# )



print(x_train.shape, x_val.shape, x_test.shape)
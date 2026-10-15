import tensorflow as tf
from tools import plot_similarity_binned
import pandas as pd

MODEL_NAME = 'models-final/sep_e_log_normalized_w_validation_cauchy_schwartz_global_2.keras'
DATA_PATH = "cleaned-dtw-SEP-EC-data"
DATA_PREFIX = 'sep_e_log_normalized'

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
    f"{DATA_PATH}/{DATA_PREFIX}",
)

model = tf.keras.models.load_model(MODEL_NAME)

plot_similarity_binned(
    model,
    x_test[::3],
    y_test[::3],
    ticks=8,
    rank_based=True,
)

plot_similarity_binned(
    model,
    x_test[::3],
    y_test[::3],
    ticks=8,
    rank_based=False,
)
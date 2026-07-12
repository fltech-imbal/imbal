import pandas as pd
import numpy as np

def load_sep_ec_data(path_prefix, use_delta=True):
    training_data = pd.read_csv(path_prefix + '_training.csv')
    test_data = pd.read_csv(path_prefix + '_test.csv')
    val_data = pd.read_csv(path_prefix + '_validation.csv')
    if use_delta:
        training_labels = training_data.pop("delta_log_Intensity")
        val_labels = val_data.pop("delta_log_Intensity")
        test_labels = test_data.pop("delta_log_Intensity")
    else:
        training_labels = training_data.pop("Proton Intensity")
        val_labels = val_data.pop("Proton Intensity")
        test_labels = test_data.pop("Proton Intensity")
    training_data = training_data.to_numpy()
    val_data = val_data.to_numpy()
    test_data = test_data.to_numpy()
    training_labels = training_labels.to_numpy()
    val_labels = val_labels.to_numpy()
    test_labels = test_labels.to_numpy()
    training_labels = training_labels.reshape(-1, 1).astype(np.float32)
    val_labels = val_labels.reshape(-1, 1).astype(np.float32)
    test_labels = test_labels.reshape(-1, 1).astype(np.float32)
    return (training_data, training_labels), (val_data, val_labels), (test_data, test_labels)
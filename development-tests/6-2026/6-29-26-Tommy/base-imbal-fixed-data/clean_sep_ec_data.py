import os, glob
import pandas as pd
import numpy as np

NORMALIZED = True
INCLUDE_CME = False
INCLUDE_PROTON = True
INCLUDE_ELECTRON = True
LOG = True
EPSILON = 1e-9
ORIGINAL_DATA_PATH = "../../../../tutorials/data/SEP-EC"

"""
Load data
"""

training_files = [f for f in glob.glob(os.path.join(ORIGINAL_DATA_PATH, "fold0/subtraining/*")) if os.path.isfile(f)]
val_files = [f for f in glob.glob(os.path.join(ORIGINAL_DATA_PATH, "fold0/validation/*")) if os.path.isfile(f)]
test_files = [f for f in glob.glob(os.path.join(ORIGINAL_DATA_PATH, "testing/*")) if os.path.isfile(f)]

def load_multiple_csv(csv_files):
    loaded_data = None
    for file in csv_files:
        df = pd.read_csv(file)
        if loaded_data is None:
            loaded_data = df
        else:
            loaded_data = pd.concat([loaded_data, df])
    return loaded_data

training_data = load_multiple_csv(training_files)
val_data = load_multiple_csv(val_files)
test_data = load_multiple_csv(test_files)

if (training_data is None) or (test_data is None) or (val_data is None):
    raise Exception("Unable to load data (specified directory is likely incorrect).")

training_labels = training_data.pop("delta_log_Intensity")
test_labels = test_data.pop("delta_log_Intensity")
val_labels = val_data.pop("delta_log_Intensity")

def clean_sep_ec_data(
    df
):
    df = df.select_dtypes(include=[np.number])
    df = df.dropna()
    df = df.drop(columns=["Event ID"])
    df = df.drop('Proton Intensity', axis=1)
    if not INCLUDE_CME:
        cme_start_index = df.columns.get_loc("Sunspot Number")
        df = df.drop(df.columns[cme_start_index:], axis=1)
    if not INCLUDE_PROTON:
        proton_start_index = df.columns.get_loc("p6.1_tminus24")
        df = df.drop(df.columns[proton_start_index:], axis=1)
    if not INCLUDE_ELECTRON:
        electron_start_index = df.columns.get_loc("e0.5_tminus24")
        electron_end_index = df.columns.get_loc("p6.1_tminus24")
        df = df.drop(df.columns[electron_start_index:electron_end_index], axis=1)

    return df

training_data = clean_sep_ec_data(training_data)
val_data = clean_sep_ec_data(val_data)
test_data = clean_sep_ec_data(test_data)

training_data = np.log(training_data + EPSILON)
val_data = np.log(val_data + EPSILON)
test_data = np.log(test_data + EPSILON)

if NORMALIZED:
    absolute_max_values = training_data.abs().max()
    test_data = test_data.div(absolute_max_values, axis=1)
    training_data = training_data.div(absolute_max_values, axis=1)
    val_data = val_data.div(absolute_max_values, axis=1)

training = pd.concat([training_data, training_labels], axis=1)
val = pd.concat([val_data, val_labels], axis=1)
test = pd.concat([test_data, test_labels], axis=1)

print(training_data.columns)
print(training_data.shape)
print(training_labels.shape)
print(val_data.shape)
print(val_labels.shape)
print(test_data.shape)
print(test_labels.shape)

training_array = training_data.to_numpy()
mins = np.min(training_array, axis=0)
maxs = np.max(training_array, axis=0)
medians = np.median(training_array, axis=0)
for i in range(len(mins)):
    print(mins[i], medians[i], maxs[i])

training.to_csv(f'SEP-E/sep_e{"" if INCLUDE_PROTON else "_no_proton"}{"" if INCLUDE_ELECTRON else "_no_electron"}{"_log" if LOG else ""}{"_normalized" if NORMALIZED else ""}_training.csv', index=False)
val.to_csv(f'SEP-E/sep_e{"" if INCLUDE_PROTON else "_no_proton"}{"" if INCLUDE_ELECTRON else "_no_electron"}{"_log" if LOG else ""}{"_normalized" if NORMALIZED else ""}_validation.csv', index=False)
test.to_csv(f'SEP-E/sep_e{"" if INCLUDE_PROTON else "_no_proton"}{"" if INCLUDE_ELECTRON else "_no_electron"}{"_log" if LOG else ""}{"_normalized" if NORMALIZED else ""}_test.csv', index=False)
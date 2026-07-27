import os, glob
import pandas as pd
import numpy as np

NORMALIZED = True
INCLUDE_CME = True
INCLUDE_PROTON = True
INCLUDE_ELECTRON = True
LOG = True
LOG_LABELS = True

EPSILON = 1e-9
LABEL_EPSILON = 1e-5
ORIGINAL_DATA_PATH = "dtw-SEP-EC-data"
USE_DELTA = False
INCLUDE_EVENT_ID = True

"""
Load data
"""

folds = [glob.glob(os.path.join(ORIGINAL_DATA_PATH, f"fold-{i+1}/*.csv")) for i in range(5)]

training_files = folds[0] + folds[1] + folds[3]
val_files = folds[4]
test_files = folds[2]

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

if USE_DELTA:
    training_labels = training_data.pop("delta_log_Intensity")
    test_labels = test_data.pop("delta_log_Intensity")
    val_labels = val_data.pop("delta_log_Intensity")
else:
    training_labels = training_data.pop("Proton Intensity")
    test_labels = test_data.pop("Proton Intensity")
    val_labels = val_data.pop("Proton Intensity")

if LOG_LABELS:
    training_labels = np.log(training_labels + LABEL_EPSILON)
    test_labels = np.log(test_labels + LABEL_EPSILON)
    val_labels = np.log(val_labels + LABEL_EPSILON)

def clean_sep_ec_data(
    df
):
    df = df.select_dtypes(include=[np.number])
    df = df.dropna()
    if INCLUDE_EVENT_ID:
        ids = df['Event ID']
    else:
        ids = None
    df = df.drop(columns=["Event ID"])
    if 'Proton Intensity' in df.columns:
        df = df.drop('Proton Intensity', axis=1)
    if 'delta_log_Intensity' in df.columns:
        df = df.drop('delta_log_Intensity', axis=1)
    if not INCLUDE_CME:
        cme_start_index = df.columns.get_loc("Sunspot Number")
        df = df.drop(df.columns[cme_start_index:], axis=1)
    if not INCLUDE_PROTON:
        proton_start_index = df.columns.get_loc("p6.1_tminus24")
        df = df.drop(df.columns[proton_start_index:], axis=1)
        df = df.drop('Proton Intensity', axis=1)
    if not INCLUDE_ELECTRON:
        electron_start_index = df.columns.get_loc("e0.5_tminus24")
        electron_end_index = df.columns.get_loc("p6.1_tminus24")
        df = df.drop(df.columns[electron_start_index:electron_end_index], axis=1)

    return df, ids

training_data, training_ids = clean_sep_ec_data(training_data)
val_data, val_ids = clean_sep_ec_data(val_data)
test_data, test_ids = clean_sep_ec_data(test_data)

if LOG:
    training_data.iloc[:, :156] = np.log(training_data.iloc[:, :156] + EPSILON)
    val_data.iloc[:, :156] = np.log(val_data.iloc[:, :156] + EPSILON)
    test_data.iloc[:, :156] = np.log(test_data.iloc[:, :156] + EPSILON)

if NORMALIZED:
    absolute_max_values = training_data.abs().max()
    test_data = test_data.div(absolute_max_values, axis=1)
    training_data = training_data.div(absolute_max_values, axis=1)
    val_data = val_data.div(absolute_max_values, axis=1)

if INCLUDE_EVENT_ID:
    training = pd.concat([training_ids, training_data, training_labels], axis=1)
    val = pd.concat([val_ids, val_data, val_labels], axis=1)
    test = pd.concat([test_ids, test_data, test_labels], axis=1)
else:
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

training.to_csv(f'cleaned-dtw-SEP-EC-data/sep_e{"c" if INCLUDE_CME else ""}{"" if INCLUDE_PROTON else "_no_proton"}{"" if INCLUDE_ELECTRON else "_no_electron"}{"_log" if LOG else ""}{"_normalized" if NORMALIZED else ""}{"_w_ids" if INCLUDE_EVENT_ID else ""}_training.csv', index=False)
val.to_csv(f'cleaned-dtw-SEP-EC-data/sep_e{"c" if INCLUDE_CME else ""}{"" if INCLUDE_PROTON else "_no_proton"}{"" if INCLUDE_ELECTRON else "_no_electron"}{"_log" if LOG else ""}{"_normalized" if NORMALIZED else ""}{"_w_ids" if INCLUDE_EVENT_ID else ""}_validation.csv', index=False)
test.to_csv(f'cleaned-dtw-SEP-EC-data/sep_e{"c" if INCLUDE_CME else ""}{"" if INCLUDE_PROTON else "_no_proton"}{"" if INCLUDE_ELECTRON else "_no_electron"}{"_log" if LOG else ""}{"_normalized" if NORMALIZED else ""}{"_w_ids" if INCLUDE_EVENT_ID else ""}_test.csv', index=False)
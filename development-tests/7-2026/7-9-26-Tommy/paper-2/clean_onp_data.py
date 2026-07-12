import os, glob
import pandas as pd
import numpy as np

NORMALIZED = True
EPSILON = 1e-9
ORIGINAL_DATA_PATH = "data"
LOG_LABELS = True
RANDOMIZED = True

"""
Load data
"""

all_data = pd.read_csv(os.path.join(ORIGINAL_DATA_PATH, 'OnlineNewsPopularity.csv'))
all_data.columns = [x.strip() for x in all_data.columns]
all_data = all_data.drop('url', axis=1)

if RANDOMIZED:
    indices = np.random.permutation(len(all_data))
    all_data = all_data.iloc[indices]

training_data = all_data.iloc[:25000]
print(training_data.columns)
val_data = all_data.iloc[25000:32000]
test_data = all_data.iloc[32000:]

if (training_data is None) or (test_data is None) or (val_data is None):
    raise Exception("Unable to load data (specified directory is likely incorrect).")

training_labels = training_data.pop("shares")
test_labels = test_data.pop("shares")
val_labels = val_data.pop("shares")

if LOG_LABELS:
    training_labels = np.log(training_labels)
    test_labels = np.log(test_labels)
    val_labels = np.log(val_labels)

def clean_sep_ec_data(
    df
):
    df = df.select_dtypes(include=[np.number])
    df = df.dropna()

    return df

training_data = clean_sep_ec_data(training_data)
val_data = clean_sep_ec_data(val_data)
test_data = clean_sep_ec_data(test_data)

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

training.to_csv(f'cleaned-ONP-data/onp{"_normalized" if NORMALIZED else ""}{"_log_labels" if LOG_LABELS else ""}{"_randomized" if RANDOMIZED else ""}_training.csv', index=False)
val.to_csv(f'cleaned-ONP-data/onp{"_normalized" if NORMALIZED else ""}{"_log_labels" if LOG_LABELS else ""}{"_randomized" if RANDOMIZED else ""}_validation.csv', index=False)
test.to_csv(f'cleaned-ONP-data/onp{"_normalized" if NORMALIZED else ""}{"_log_labels" if LOG_LABELS else ""}{"_randomized" if RANDOMIZED else ""}_test.csv', index=False)
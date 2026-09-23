import pandas as pd
import numpy as np
import os, glob

# DATA_PATH = "base-imbal-fixed-data/cleaned-dtw-SEP-EC-data"
# DATA_PREFIX = 'sep_ec_log_normalized'
ORIGINAL_DATA_PATH = "dtw-data"
TOLERANCE = 1e-2

def load_multiple_csv(csv_files):
    loaded_data = None
    for file in csv_files:
        df = pd.read_csv(file)
        if loaded_data is None:
            loaded_data = df
        else:
            loaded_data = pd.concat([loaded_data, df])
    return loaded_data

folds = [glob.glob(os.path.join(ORIGINAL_DATA_PATH, f"fold-{i+1}/*.csv")) for i in range(5)]

training_files = folds[0] + folds[1] + folds[3]
val_files = folds[4]
test_files = folds[2]

all = folds[0] + folds[1] + folds[2] + folds[3] + folds[4]


def test_deltas(file):
    print(file)
    df = pd.read_csv(file)
    manual_deltas = np.log(df['Proton Intensity'] + 1e-9) - np.log(df['p_t'] + 1e-9)

    errors = np.abs(df['delta_log_Intensity'] - manual_deltas).to_numpy()
    # print(errors)
    sort_indices = np.argsort(errors)
#     print(errors[sort_indices])
#     print(df['delta_log_Intensity'].to_numpy()[sort_indices[-1]])
#     print(manual_deltas.to_numpy()[sort_indices[-1]])
    print(len(errors[errors > TOLERANCE]))
    return len(errors[errors > TOLERANCE]), len(errors)

total_bad, total = 0, 0
for file in all:
    one, two =  test_deltas(file)
    total_bad += one
    total += two

print(total)
print(total_bad)
#
# training_data = load_multiple_csv(training_files)
# val_data = load_multiple_csv(val_files)
# test_data = load_multiple_csv(test_files)
#
# print(training_data['delta_log_Intensity'])
# print(training_data['Proton Intensity'] - training_data['p_t'])
# # print(np.log(training_data['Proton Intensity']) - np.log(training_data['p_t']))
#
# manual_deltas = np.log(training_data['Proton Intensity'] + 1e-9) - np.log(training_data['p_t'] + 1e-9)
#
# errors = np.abs(training_data['delta_log_Intensity'] - manual_deltas).to_numpy()
# print(errors)
# sort_indices = np.argsort(errors)
# print(errors[sort_indices])
# print(training_data['delta_log_Intensity'].to_numpy()[sort_indices[-1]])
# print(manual_deltas.to_numpy()[sort_indices[-1]])
# print(len(errors) - len(errors[errors < 1e-4]))
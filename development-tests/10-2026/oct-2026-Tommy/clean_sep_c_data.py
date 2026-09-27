import os, glob
import pandas as pd
import numpy as np

ORIGINAL_DATA_PATH = "data/SEP-C"

"""
Load data
"""

training = pd.read_csv(os.path.join(ORIGINAL_DATA_PATH, "sep_10mev_training.csv"))
val = pd.read_csv(os.path.join(ORIGINAL_DATA_PATH, "sep_10mev_validation.csv"))
test = pd.read_csv(os.path.join(ORIGINAL_DATA_PATH, "sep_10mev_test.csv"))

training_array = training.to_numpy()
mins = np.min(training_array, axis=0)
maxs = np.max(training_array, axis=0)
medians = np.median(training_array, axis=0)
for i in range(len(mins)):
    print(mins[i], medians[i], maxs[i])

training.to_csv(f'cleaned-SEP-C-data/sep_c_training.csv', index=False)
val.to_csv(f'cleaned-SEP-C-data/sep_c_validation.csv', index=False)
test.to_csv(f'cleaned-SEP-C-data/sep_c_test.csv', index=False)
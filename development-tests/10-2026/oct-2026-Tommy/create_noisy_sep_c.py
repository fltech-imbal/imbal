from tools import load_sep_c_data
import pandas as pd
import numpy as np

data_path = "cleaned-SEP-C-data"
path_prefix = 'sep_c'
noise_range = 0.01


training_data = pd.read_csv(data_path + '/' + path_prefix + '_training.csv')
test_data = pd.read_csv(data_path + '/' + path_prefix + '_test.csv')
val_data = pd.read_csv(data_path + '/' + path_prefix + '_validation.csv')

training_data['ln_peak_intensity'] += np.random.normal(-noise_range,noise_range,len(training_data))
val_data['ln_peak_intensity'] += np.random.normal(-noise_range,noise_range,len(val_data))
test_data['ln_peak_intensity'] += np.random.normal(-noise_range,noise_range,len(test_data))
training_data.to_csv(data_path + '/' + path_prefix + '_w_noise_training.csv', index=False)
val_data.to_csv(data_path + '/' + path_prefix + '_w_noise_validation.csv', index=False)
test_data.to_csv(data_path + '/' + path_prefix + '_w_noise_test.csv', index=False)


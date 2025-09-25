import os

import imbal.regression.generate_weights
from imbal.classification import generate_weights as generate_classification_weights
from imbal.regression import generate_weights as generate_regression_weights
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import csv

def read_csv_to_list_of_lists(filepath):
    data = []
    with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            data.append(row)
    return data

# Download latest version
# path = kagglehub.dataset_download("arashnic/imbalanced-data-practice")
# data = np.array(read_csv_to_list_of_lists(path + '/aug_train.csv'))
# data = data[1:1000, 2].astype(np.float64)
# print(data)

labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2])

print(generate_classification_weights(labels, {
    0: 1/6,
    1: 1/3,
    2: 1/2
}).tolist())

print(generate_classification_weights(labels).tolist())

# data = np.array(read_csv_to_list_of_lists('/mnt/c/Users/tommy/Desktop/Repos/dr-chan-work-demo/CISIR-data/SARCOS/sarcos_inv_training.csv'))
# print(data.shape)
# data = data[1:, -1].astype(float)

# data = np.array(read_csv_to_list_of_lists('/mnt/c/Users/tommy/Desktop/Repos/dr-chan-work-demo/CISIR-data/SEP-C/sep_10mev_training.csv'))
# print(data.shape)
# data = data[1:, 22].astype(float)

print(os.getcwd())
data = np.array(read_csv_to_list_of_lists(f'/mnt/c/Users/tommy/Desktop/Repos/dr-chan-work-demo/CISIR-data/SEP-EC/training/sep_event_1_filled_ie_trim.csv'))[1:]
for i in range(43):
    if os.path.exists(f'/mnt/c/Users/tommy/Desktop/Repos/dr-chan-work-demo/CISIR-data/SEP-EC/training/sep_event_{i+2}_filled_ie_trim.csv'):
        data = np.concatenate([data, read_csv_to_list_of_lists(f'/mnt/c/Users/tommy/Desktop/Repos/dr-chan-work-demo/CISIR-data/SEP-EC/training/sep_event_{i+2}_filled_ie_trim.csv')[1:]])
print(data.shape)
data = data[:, 182].astype(float)
print(data.shape)

bins = 30
weights, kde = generate_regression_weights(data, return_kde=True, bin_count=bins, visualize_kde=True)
# print(weights.tolist())
import os
from imbal.classification import generate_weights as generate_classification_weights
from imbal.regression import generate_weights as generate_regression_weights
import numpy as np
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

PATH_START = '/mnt/c/Users/tommy/Desktop/Repos/dr-chan-work-demo'
print(os.getcwd())

# data = np.array(read_csv_to_list_of_lists(f'{PATH_START}/CISIR-data/SARCOS/sarcos_inv_training.csv'))
# print(data.shape)
# data = data[1:, -1].astype(float)

# data = np.array(read_csv_to_list_of_lists(f'{PATH_START}/CISIR-data/SEP-C/sep_10mev_training.csv'))
# print(data.shape)
# data = data[1:, 22].astype(float)

data = np.array(read_csv_to_list_of_lists(f'{PATH_START}/CISIR-data/SEP-EC/training/sep_event_1_filled_ie_trim.csv'))[1:]
for i in range(43):
    if os.path.exists(f'{PATH_START}/CISIR-data/SEP-EC/training/sep_event_{i+2}_filled_ie_trim.csv'):
        data = np.concatenate([data, read_csv_to_list_of_lists(f'{PATH_START}/CISIR-data/SEP-EC/training/sep_event_{i+2}_filled_ie_trim.csv')[1:]])
print(data.shape)
data = data[:, 182].astype(float)
print(data.shape)


BINS=128
from matplotlib import pyplot as plt

# print(np.std(data.reshape(-1,)))

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(7*4, 5))

weights, kde = generate_regression_weights(
    data,
    return_kde=True,
    bandwidth='binned',
    optimization='linear_interpolation',
    use_axes=ax[0],
    bin_count=BINS
)

weights, kde = generate_regression_weights(
    data,
    return_kde=True,
    bandwidth='binned_fit',
    optimization='linear_interpolation',
    use_axes=ax[1],
    bin_count=BINS
)

weights, kde = generate_regression_weights(
    data,
    return_kde=True,
    bandwidth='scott',
    optimization='linear_interpolation',
    use_axes=ax[2],
    bin_count=BINS
)

weights, kde = generate_regression_weights(
    data,
    return_kde=True,
    bandwidth='silverman',
    optimization='linear_interpolation',
    use_axes=ax[3],
    bin_count=BINS
)

plt.savefig('high-sep-ec-ours-new-scott-silverman.png')
plt.show()


# bandwidth_min = 0.02
# bandwidth_max = 0.2
# bandwidth_steps = 10
# bin_min = 3
# bin_max = 7
#
# fig, axes = plt.subplots(
#     nrows=bandwidth_steps,
#     ncols=bin_max - bin_min + 1,
#     figsize=(7*(bin_max - bin_min + 1), bandwidth_steps*5)
# )
# for i in range(bandwidth_steps):
#     for j in range(bin_max - bin_min + 1):
#         current_bins = 2 ** (bin_min + j)
#         current_bandwidth = bandwidth_min + i * (bandwidth_max - bandwidth_min) / (bandwidth_steps - 1)
#         print(current_bandwidth, current_bins)
#         weights, kde = generate_regression_weights(
#             data,
#             return_kde=True,
#             # visualize_kde=True,
#             bandwidth=current_bandwidth,
#             # verbose=True,
#             optimization='linear_interpolation',
#             bin_count=current_bins,
#             save_figure=axes[i, j]
#         )
#
# plt.savefig('3.png')
# plt.show()


# print(weights[:100])
# exact_weights, kde = generate_regression_weights(
#     data,
#     return_kde=True,
#     bandwidth='binned',
#     bin_count=BINS
# )
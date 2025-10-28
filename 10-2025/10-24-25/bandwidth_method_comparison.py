import os
import numpy as np
import csv

from sklearn.neighbors import KernelDensity

import imbal


def read_csv_to_list_of_lists(filepath):
    data = []
    with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            data.append(row)
    return data

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

import matplotlib.pyplot as plt
import time

BINS=64

print(1)
kl_bandwidth = imbal.regression.kde.fit_kde(
    data,
    fit_method='kl_divergence',
    bin_count=BINS,
    steps_per_bin=5
)
scott_bandwidth = imbal.regression.kde.fit_kde(
    data,
    fit_method='scott',
)
silverman_bandwidth = imbal.regression.kde.fit_kde(
    data,
    fit_method='silverman',
)
fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(16,4))

kl_kde = KernelDensity(bandwidth=kl_bandwidth).fit(data.reshape(-1, 1))
imbal.regression.plot_kde_1d(
    data,
    kl_kde,
    bin_count=BINS,
    use_axes=ax[0]
)
ax[0].set_title(f'kl_divergence method (bandwidth={kl_bandwidth:.3f})')
scott_kde = KernelDensity(bandwidth=scott_bandwidth).fit(data.reshape(-1, 1))
imbal.regression.plot_kde_1d(
    data,
    scott_kde,
    bin_count=BINS,
    use_axes=ax[1]
)
ax[1].set_title(f'scott method (bandwidth={scott_bandwidth:.3f})')
silverman_kde = KernelDensity(bandwidth=silverman_bandwidth).fit(data.reshape(-1, 1))
imbal.regression.plot_kde_1d(
    data,
    silverman_kde,
    bin_count=BINS,
    use_axes=ax[2]
)
ax[2].set_title(f'silverman method (bandwidth={silverman_bandwidth:.3f})')

plt.savefig('bandwidth_method_comparison-dataset_3.png')
plt.show()
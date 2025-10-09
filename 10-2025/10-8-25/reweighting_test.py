import os
import numpy as np
import kagglehub
import csv
import imbal


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

BINS=32
print(1)
kde = imbal.regression.kde.fit_kde(
    data,
    bandwidth='kl_divergence',
    bin_count=BINS
)
print(2)
start = time.time()
densities = imbal.regression.get_densities(
    data,
    kde,
    distribution_samples=10*BINS
)
end = time.time()
print('normal', end-start)
print(densities.shape)
print(3)
start = time.time()
lin_int_densities, lin_int_approx = imbal.regression.get_densities(
    data,
    kde,
    distribution_samples=10*BINS,
    optimization='linear_interpolation',
    return_optimization=True
)
end=time.time()
print('lin_int', end-start)
print(lin_int_densities.shape)
print(4)
start=time.time()
loc_approx_densities, loc_approx_approx = imbal.regression.get_densities(
    data,
    kde,
    distribution_samples=10*BINS,
    optimization='local_approximation',
    k=BINS*10,
    precision=1e-6,
    return_optimization=True
)
end=time.time()
print('loc_approx', end-start)
print(loc_approx_densities.shape)
print(5)
num_bins = 32  # Number of desired logarithmic bins
lin_errors = np.abs(lin_int_densities - densities).reshape(-1)
bins = np.logspace(np.log10(min(lin_errors)), np.log10(max(lin_errors)), num_bins + 1)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), constrained_layout=True)
ax[0].hist(lin_errors, bins=bins, alpha=0.6)
ax[0].set_xscale('log')
ax[0].set_title('Errors of linear_interpolation')
ax[0].set_ylim(auto=True)
loc_approx_errors = np.abs(loc_approx_densities - densities).reshape(-1)
ax[1].hist(loc_approx_errors, bins=bins, alpha=0.6)
ax[1].set_xscale('log')
ax[1].set_title('Errors of local_approximation')
ax[1].set_ylim(auto=True)
plt.savefig('sep-ec-optimization-error-hists.png')
plt.show()

print('lin_int average error:', np.mean(lin_errors))
print('loc_approx average error:', np.mean(loc_approx_errors))

###################### OR ########################

# imbal.regression.labels_to_kde_weights(
#     data,
#     bandwidth='kl_divergence',
#     bin_count=BINS,
#     optimization='linear_interpolation',
#     plot_kde=True
# )


# from matplotlib import pyplot as plt
#
# fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(7*4, 5))
#
# weights, kde = generate_regression_weights(
#     data,
#     return_kde=True,
#     bandwidth='binned',
#     optimization='linear_interpolation',
#     use_axes=ax[0],
#     bin_count=BINS
# )
#
# weights, kde = generate_regression_weights(
#     data,
#     return_kde=True,
#     bandwidth='binned_fit',
#     optimization='linear_interpolation',
#     use_axes=ax[1],
#     bin_count=BINS
# )
#
# weights, kde = generate_regression_weights(
#     data,
#     return_kde=True,
#     bandwidth='scott',
#     optimization='linear_interpolation',
#     use_axes=ax[2],
#     bin_count=BINS
# )
#
# weights, kde = generate_regression_weights(
#     data,
#     return_kde=True,
#     bandwidth='silverman',
#     optimization='linear_interpolation',
#     use_axes=ax[3],
#     bin_count=BINS
# )
#
# plt.savefig('high-sep-ec-ours-new-scott-silverman.png')
# plt.show()
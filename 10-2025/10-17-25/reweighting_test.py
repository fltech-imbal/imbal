import os
import numpy as np
import kagglehub
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

# Download latest version
# path = kagglehub.dataset_download("arashnic/imbalanced-data-practice")
# data = np.array(read_csv_to_list_of_lists(path + '/aug_train.csv'))
# data = data[1:1000, 2].astype(np.float64)
# print(data)

PATH_START = '/mnt/c/Users/tommy/PycharmProjects/DrChanWorkPlayground'
print(os.getcwd())

data = np.array(read_csv_to_list_of_lists(f'{PATH_START}/CISIR-data/SARCOS/sarcos_inv_training.csv'))
print(data.shape)
data = data[1:, -1].astype(float)

# data = np.array(read_csv_to_list_of_lists(f'{PATH_START}/CISIR-data/SEP-C/sep_10mev_training.csv'))
# print(data.shape)
# data = data[1:, 22].astype(float)

# data = np.array(read_csv_to_list_of_lists(f'{PATH_START}/CISIR-data/SEP-EC/training/sep_event_1_filled_ie_trim.csv'))[1:]
# for i in range(43):
#     if os.path.exists(f'{PATH_START}/CISIR-data/SEP-EC/training/sep_event_{i+2}_filled_ie_trim.csv'):
#         data = np.concatenate([data, read_csv_to_list_of_lists(f'{PATH_START}/CISIR-data/SEP-EC/training/sep_event_{i+2}_filled_ie_trim.csv')[1:]])
# print(data.shape)
# data = data[:, 182].astype(float)
# print(data.shape)

import matplotlib.pyplot as plt
import time

BINS=32

print(1)
bandwidth = imbal.regression.kde.fit_kde(
    data,
    bandwidth='kl_divergence',
    bin_count=BINS,
    steps_per_bin=5
)
kde = KernelDensity(bandwidth=bandwidth).fit(data.reshape(-1, 1))
imbal.regression.plot_kde_1d(
    data,
    kde,
    bin_count=BINS,
    save_figure='dataset-1.png'
)
print(2)
start = time.time()
densities = imbal.regression.get_densities(
    data,
    bandwidth
)
end = time.time()
print('normal', end-start)
print(densities.shape)
print(3)
start = time.time()
lin_int_densities, lin_int_approx = imbal.regression.get_densities(
    data,
    bandwidth,
    interpolation_samples=10*BINS,
    interpolation_method='linear',
    return_interpolation_samples=True
)
end=time.time()
print('lin_int', end-start)
print(lin_int_densities.shape)
print(4)
start=time.time()
loc_approx_densities, loc_approx_approx = imbal.regression.get_densities(
    data,
    bandwidth,
    interpolation_samples=10*BINS,
    # optimization='local_approximation',
    atol=1e-4,
    return_interpolation_samples=True
)
end=time.time()
print('loc_approx', end-start)
print(loc_approx_densities.shape)

print(4.5)
start=time.time()
atol_densities = imbal.regression.get_densities(
    data,
    bandwidth,
    atol=1e-4
)
end=time.time()
print('atol', end-start)
print(atol_densities.shape)
print(5)
start=time.time()
no_k_densities = imbal.regression.get_densities(
    data,
    bandwidth,
    interpolation_samples=10*BINS,
    # optimization='local_approximation',
    # k=BINS*10,
    atol=1e-4
)
end=time.time()
print('no_k', end-start)
print(no_k_densities.shape)
print(6)
num_bins = 32  # Number of desired logarithmic bins
lin_errors = np.abs(lin_int_densities - densities).reshape(-1)
bins = np.logspace(np.log10(min(lin_errors)), np.log10(max(lin_errors)), num_bins + 1)
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4), constrained_layout=True)
ax[0].hist(lin_errors, bins=bins, alpha=0.6)
ax[0].set_xscale('log')
ax[0].set_title('Errors of linear_interpolation')
ax[0].set_ylim(auto=True)
loc_approx_errors = np.abs(loc_approx_densities - densities).reshape(-1)
bins = np.linspace(0, max(loc_approx_errors), num_bins + 1)
ax[1].hist(loc_approx_errors, bins=bins, alpha=0.6)
# ax[1].set_xscale('log')
ax[1].set_title('Errors of local_approximation')
ax[1].set_ylim(auto=True)
atol_errors = np.abs(atol_densities - densities).reshape(-1)
bins = np.linspace(0, max(atol_errors), num_bins + 1)
ax[2].hist(atol_errors, bins=bins, alpha=0.6)
# ax[2].set_xscale('log')
ax[2].set_title('Errors of atol approximation')
ax[2].set_ylim(auto=True)
no_k_errors = np.abs(no_k_densities - densities).reshape(-1)
bins = np.linspace(0, max(no_k_errors), num_bins + 1)
ax[3].hist(no_k_errors, bins=bins, alpha=0.6)
# ax[3].set_xscale('log')
ax[3].set_title('Errors of no_k approximation')
ax[3].set_ylim(auto=True)
plt.savefig('dataset-1-error-histogram.png')
plt.show()

densities = densities.reshape(-1)

print('lin_int average error:', np.mean(lin_errors))
print('loc_approx average error:', np.mean(loc_approx_errors))
print('atol average error:', np.mean(atol_errors))
print('no_k average error:', np.mean(no_k_errors))
print('lin_int average % error:', np.mean(lin_errors / densities * 100))
print('loc_approx average % error:', np.mean(loc_approx_errors / densities * 100))
print('atol average % error:', np.mean(atol_errors / densities * 100))
print('no_k average % error:', np.mean(no_k_errors / densities * 100))

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
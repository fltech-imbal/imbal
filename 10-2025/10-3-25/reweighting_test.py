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


BINS=48

# kde = imbal.regression.kde.fit_kde(
#     data,
#     bandwidth='kl_divergence',
#     bin_count=BINS
# )
#
# weights, approx = imbal.regression.generate_weights(
#     data,
#     density_mapping=kde,
#     bin_count=BINS,
#     optimization='linear_interpolation',
#     return_optimization=True
# )
#
# imbal.regression.kde.plot_kde(
#     data,
#     kde,
#     bin_count=BINS,
#     approximation=approx
# )

###################### OR ########################

imbal.regression.helpers.labels_to_kde_weights(
    data,
    bandwidth='kl_divergence',
    bin_count=BINS,
    optimization='linear_interpolation',
    plot_kde=True
)


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
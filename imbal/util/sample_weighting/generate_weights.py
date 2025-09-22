import numpy as np
from sklearn.neighbors import KernelDensity


def generate_regression_weights(
        labels,
        bandwidth=1,
        bins=10,
        weight_mapping=None
    ):
    exp = np.vectorize(lambda x: 10 ** x)

    if bandwidth == 'binned':
        labels = labels.reshape(-1,)
        labels = np.sort(labels)
        label_min = labels[0]
        label_max = labels[-1] + 1e-6
        step = (label_max - label_min) / bins
        bin_contents = []
        for i in range(bins):
            current_bin = labels[(labels >= i*step) & (labels < (i+1)*step)]
            bin_contents.append(current_bin)
        column_labels = labels.reshape(-1, 1)
        temp_kde = KernelDensity(bandwidth=label_max-label_min)
        temp_kde.fit(column_labels)
        temp_densities = [1 / exp(temp_kde.score_samples(values.reshape(-1, 1)).reshape(-1,)) if values.shape[0] > 0 else np.array([]) for values in bin_contents]
        density_averages = [sum(values) / len(values) if len(values) > 0 else 0 for values in temp_densities]
        print(density_averages)


        bandwidth = 1


    kde = KernelDensity(bandwidth=bandwidth)
    labels = labels.reshape(-1, 1)
    kde.fit(labels)
    reweights = 1 / exp(kde.score_samples(labels).reshape(-1,))
    reweights = reweights / np.sum(reweights)
    if weight_mapping is not None:
        reweights = np.vectorize(weight_mapping)(reweights)
        reweights = reweights / np.sum(reweights)

    return reweights
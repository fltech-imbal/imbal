import numpy as np
from sklearn.neighbors import KernelDensity

def generate_classification_weights(labels, weight_mapping=None):

    labels = labels.reshape(-1,)
    unique_classes, unique_counts = np.unique(labels, return_counts=True)
    full_weight_mapping = {}
    balanced_mapping = {}
    weight_sum = 0

    if isinstance(weight_mapping, dict):
        for cls in unique_classes:
            if cls in weight_mapping:
                full_weight_mapping[cls] = weight_mapping[cls]
            else:
                full_weight_mapping[cls] = 1
            weight_sum += full_weight_mapping[cls]
    else:
        if weight_mapping is None:
            for cls in unique_classes:
                full_weight_mapping[cls] = 1
                weight_sum += full_weight_mapping[cls]
        else:
            if len(weight_mapping) != len(unique_classes):
                raise ValueError('When passing weights as a list, the length of the list of weights must be equal to the number of classes.')
            for cls, weight in zip(unique_classes, weight_mapping):
                full_weight_mapping[cls] = weight
                weight_sum += weight



    for label, count  in zip(unique_classes, unique_counts):
        balanced_mapping.update({label: full_weight_mapping[label] / weight_sum / count})

    return np.array([balanced_mapping[label] for label in labels])

def generate_regression_weights(
        labels,
        bandwidth=1,
        bins=10,
        weight_distribution=None
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
    if weight_distribution is not None:
        reweights = np.vectorize(weight_distribution)(reweights)
        reweights = reweights / np.sum(reweights)

    return reweights
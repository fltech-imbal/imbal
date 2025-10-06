import numpy as np
from sklearn.neighbors import KernelDensity
from math import sqrt, pi, log, floor
from imbal.util.sample_weighting import get_label_bin_bounds, calculate_bin_count

def generate_weights(
        labels,
        density_mapping=None,
        optimization=None,
        steps_per_bin=10,
        bin_count=None,
        average_samples_per_bin=100,
        padding_factor=0.01,
        return_optimization=False
    ):
    """

    Args:
        return_optimization:
        average_samples_per_bin:
        labels:
        bin_count:
        density_mapping:
        steps_per_bin:
        padding_factor: Optional, default :code:`0.01`. Used to add a small padding to
            the data range used for binning. This padding should be specified as a percentage
            of the range of the data labels. This padding allows for a more graceful
            handling of scenarios where the minimum or maximum of the labels is the most
            frequent value in the labels.
        optimization:

    Returns:

    """

    approx = None

    bin_count = calculate_bin_count(labels, bin_count, average_samples_per_bin)

    if isinstance(density_mapping, KernelDensity):
        # Use KDE estimation to generate weights
        if optimization is None:
            reweights = 1 / np.exp(density_mapping.score_samples(labels.reshape(-1, 1)).reshape(-1, ))
        elif optimization == 'linear_interpolation':
            sample_points, sample_densities = _linearly_interpolate_kde(
                labels,
                density_mapping,
                bin_count,
                steps_per_bin,
                padding_factor
            )
            approx = (sample_points, sample_densities)
            sample_densities = 1 / sample_densities
            reweights = np.interp(labels, sample_points, sample_densities)
        elif optimization == 'local':
            points, densities = _local_kde_optimization(labels, density_mapping, steps_per_bin)
            approx = (points, densities)
            reweights = 1 / densities
        else:
            raise ValueError("'optimization' must be either 'linear_interpolation', 'local', or None")

    elif isinstance(density_mapping, list):
        # Use uniform mapping for weights
        reweights = np.array(density_mapping) / np.sum(density_mapping)
    else:
        # Use function mapping for weights
        vectorized_function = np.vectorize(density_mapping)
        reweights = vectorized_function(labels)
        reweights = reweights / np.sum(reweights)

    if return_optimization:
        return reweights, approx
    else:
        return reweights

def _local_kde_optimization(
        labels,
        kde,
        steps_per_bin
):
    labels = np.sort(labels.reshape(-1, ))
    bandwidth = kde.bandwidth_
    inverse_gaussian = lambda x: sqrt(-2 * log(x * (bandwidth * sqrt(2 * pi)))) * bandwidth
    delta = inverse_gaussian(1e-4 / labels.shape[0])
    k = max(2, steps_per_bin*round(log(labels.shape[0])))

    sample_densities = []
    labels = labels.reshape(-1,)
    low_index = 0
    high_index = 0
    for label in labels:
        while labels[low_index] < label - delta:
            low_index += 1
        while high_index < labels.shape[0] and labels[high_index] < label + delta:
            high_index += 1

        if high_index - low_index > k:
            stride = floor((high_index - low_index) / k)
            samples = labels[low_index:high_index:stride]
        else:
            samples = labels[low_index:high_index]

        current_kde = KernelDensity(bandwidth=bandwidth)
        current_kde.fit(samples.reshape(-1,1))
        sample_densities.append(np.exp(current_kde.score_samples(np.array([[label]]))) * (high_index - low_index) / labels.shape[0])
    sample_densities = np.array(sample_densities).reshape(-1,)
    return labels, sample_densities

def _linearly_interpolate_kde(
        labels,
        kde,
        bin_count,
        steps_per_bin,
        padding_factor
):
    labels = np.sort(labels.reshape(-1, ))
    total_samples = bin_count * steps_per_bin
    label_min, label_max, step = get_label_bin_bounds(labels, bin_count, padding_factor)
    step /= steps_per_bin
    sample_points = np.array([label_min + i*step for i in range(total_samples + 1)])
    sample_densities = np.exp(kde.score_samples(sample_points.reshape(-1,1)).reshape(-1,))
    return sample_points, sample_densities
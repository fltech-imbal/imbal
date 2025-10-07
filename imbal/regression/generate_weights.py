import numpy as np
from sklearn.neighbors import KernelDensity
from math import sqrt, pi, log, floor
from imbal.util.sample_weighting import get_label_bin_bounds, calculate_bin_count

def generate_weights(
        labels,
        density_mapping,
        optimization=None,
        steps_per_bin=10,
        bin_count=None,
        average_samples_per_bin=100,
        padding_factor=0.01,
        return_optimization=False
    ):
    """
    Generates a list of weights, where the index of each weight corresponds to the label
    at the index of the provides list of labels. The sum of all weights in the returned
    list of weights will be normalized to 1.

    Args:
        labels: A NumPy array of labels, arranged as a column vector
        density_mapping: A scikit-learn :code:`KernelDensity` instance, list, or function.
            If a :code:`KernelDensity` instance, weights will be calculated
            as the reciprocal of each points' sampled density, then normalized to 1. If a
            list, weights will be calculated as the reciprocal of each provided density, then
            normalized to 1. If a function, weights will be calculated as the reciprocal of the
            result of each points' value after being inputted to the function, then normalized
            to 1.
        optimization: Optional, default :code:`None`. For KDE sampling only. Determines the
            method that should be used to optimize density sampling from KDE. Allowed values
            are :code:`'linear_interpolation'` and :code:`'local'`. When set to :code:`'linear_interpolation'`,
            an approximation of the KDE curve is made by sampling a number of evenly distributed
            points along the curve equal to :code:`bin_count * steps_per_bin`, which is then used
            to sample densities. When set to :code:`local`, for each point, only the points close
            to the point being sample are used to determine the KDE value, reducing the amount of
            points used for each KDE calculation, while introducing a small error (less than :code:`1e-4`).
            If set to :code:`None`, no optimization methods are used.
        return_optimization: Optional, default :code:`False` If set to true, returns a tuple
            containing the list of x and y coordinates used to generate the optimized KDE. Mainly
            used for visualization.
        average_samples_per_bin: Optional, default :code:`100`. For KDE sampling only. Determines the
            number of bins used for histogram-based KDE approximation by the number of datapoints. For
            example, a dataset with 14500 data points with :code:`average_samples_per_bin` set to :code:`100`
            will have 145 bins.
        bin_count: Optional, default :code:`None`. For KDE sampling only. The number of bins that
            should be used for the histogram-based KDE approximation.
            If set, overrides :code:`average_samples_per_bin`.
        steps_per_bin: Optional, default :code:`10`. For KDE sampling only. Determines the number of
            steps per bin that should be used for KDE optimizations.
        padding_factor: Optional, default :code:`0.01`. Used to add a small padding to
            the data range used for binning. This padding should be specified as a percentage
            of the range of the data labels. This padding allows for a more graceful
            handling of scenarios where the minimum or maximum of the labels is the most
            frequent value in the labels.

    Returns:
        A list of weights, normalized to 1.
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
        reweights = 1 /np.array(density_mapping)
        reweights = reweights / np.sum(reweights)
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
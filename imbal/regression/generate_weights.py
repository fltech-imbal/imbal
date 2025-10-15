import numpy as np
from math import sqrt, pi, log, ceil

from sklearn.neighbors import KernelDensity

from imbal.util.sample_weighting import get_label_bin_bounds
from scipy.interpolate import RegularGridInterpolator

def get_densities(
        labels,
        bandwidth,
        optimization=None,
        distribution_samples=None,
        k=None,
        atol=0,
        padding_factor=0.01,
        return_optimization=False
):
    r"""
    This function exists as a means of extracting densities from a scitkit-learn
    :code:`KernelDensity` object. As a part of this function, some approaches to
    optimize time efficiency for KDE sampling have been included, which can
    cost at the cost of a small error in the desnity values sampled.

    Kernel density estimatinon is defined as:

    .. math::

       \hat{f}_h(x)=\frac{1}{n}\sum_{i=1}^{n}K_h(x-x_i)

    By definition, to compute the KDE of a single point, we require a summation of the
    density kernel centered at each known data point. Therefore, computing the KDE
    for a single label is an :math:`O(n)` operation, and computing the KDE for each
    label is an :math:`O(n^2)` operation. This means retreiving the densities
    for a large dataset will take a long time. To combat this, :code:`get_densities`
    implements two optimization methods.

    The first optimization method is :code:`linear_interpolation`. This method samples
    a fixed amount of points from the KDE curve (see :code:`distribution_samples` below),
    then linearly interpolates the values for each of the desired label densities. This
    linear interpolation, for a single point, is an :math:`O(\text{log}p)` operation, where
    :math:`p` is the number of points sampled to generate the linear approximation. Therefore, the
    time complexity for sampling all label densities is :math:`O(n\text{log}p)`. However, to generate
    the linear approximation is an :math:`O(n)` operation for each of the :math:`p` points, making the
    true time complexity :math:`O(pn)`.

    The second optimization available is :code:`local_approximation`. With this optimization,
    for each label, only the labels that are close to current label are considered in the KDE
    approximation. A label is considered "close" to another if its kernel affects the other by
    a value greater than :code:`precision/n`, where :code:`n` is the number of labels
    (see :code:`precision` below). To futher optimize, in the case where there are more than
    k close points (see :code:`k` below), a stride is applied to the sampling of close points to
    ensure the total number of points used for the approximation is less than or equal to :code:`k`.
    This local approximation, for a single point, is an :math:`O(k)` operation, where
    :math:`p` is the number of points sampled to generate the linear approximation. Therefore, the
    time complexity for sampling all label densities is :math:`O(kn)`.

    Args:
        labels: A NumPy array of labels, arranged as a column vector
        kde: A scikit-learn :code:`KernelDensity` object instance. Densities will be
            sampled from the object using :code:`kde.score_samples`.
        optimization: Optional, default :code:`None`. The
            method that should be used to optimize density sampling from KDE. Allowed values
            are :code:`'linear_interpolation'` and :code:`'local_approximation'`. When set to :code:`'linear_interpolation'`,
            an approximation of the KDE curve is made by sampling a number of evenly distributed
            points along the curve (see :code:`distribution_samples`), which is then used
            to sample densities. When set to :code:`local_approximation`, for each point, only the points close
            to the point being sample are used to determine the KDE value, reducing the amount of
            points used for each KDE calculation, while introducing a small error (see :code:`precision`).
            Additionally, when the number of local points is greater than some threshold (see :code:`k`),
            a fraction of those points are sampled, then the resultant density scaled accordingly.
            If set to :code:`None`, no optimization methods are used.
        distribution_samples: Optional, default :code:`None`. Used only when :code:`optimization` is set to
            :code:`linear_interpolation`. The number of points to be sampled from the density distribution.
            When set to :code:`None`, this value is computed as :code:`labels.shape[0] / 10`. If KDE was generated
            using :code:`imbal.regression.fit_kde`, :code:`steps_per_bin*bin_count` tends to be a good
            value for :code:`distribution_samples`.
        k: Optional, default :code:`None`. Used only when :code:`optimization` is set to
            :code:`local_approximation`. The maximum number of points to sample during local
            approximation. Note that because a stride method is used to sample points that match the
            local distribution of labels, the number of points being sampled in approximations where
            the number of local points is greater than k could be as little as k/2. If set to :code:`None`,
            this value is computed as :code:`10*log(labels.shape[0])`.
        precision: Optional, default :code:`1e-4`. Used only when :code:`optimization` is set to
            :code:`local_approximation`. The maximum allowed error from each sample in the
            local approximation.
        return_optimization: Optional, default :code:`False` If set to true, returns a tuple
            containing the list of x and y coordinates used to generate the optimized KDE. Mainly
            used for visualization.
        padding_factor: Optional, default :code:`0.01`. Used to add a small padding to
            the data range used for binning for the histogram. There are some instances where many datapoints in
            a dataset fall on the maximum or minimum. When viewed visually, the peak of the found KDE curve may
            appear to be on the edge, or slightly outside, of its corresponding bin (due to limited
            pixel resolution when plotting), which is undesirable for visual comparison. By padding, we can slightly increase
            the width of the histogram bins, shifting their bounds and allowing these peaks to appear
            inside the bins instead.

    Returns:
        A NumPy array of densities, arranged as a column vector

    Example:

     .. code-block:: python

        >>> labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2])
        >>> kde = imbal.regression.fit_kde(labels, bin_count=3)
        >>> densities = imbal.regression.get_densities(labels, kde)

        >>> print(densities)
        [0.674, 0.674, 0.674, 0.674, 0.674, 0.674, 0.674, 0.674, 0.674, 0.674,
         0.674, 0.674, 0.349, 0.349, 0.349, 0.349, 0.349, 0.349, 0.118, 0.118]
    """
    if distribution_samples is None:
        distribution_samples = round(labels.shape[0] / 10)

    labels = labels.reshape(labels.shape[0], -1)

    kde = KernelDensity(bandwidth=bandwidth, atol=atol).fit(labels)

    # Use KDE estimation to generate weights
    points, densities = None, None
    if optimization is None:
        densities = np.exp(kde.score_samples(labels).reshape(-1, ))
    elif optimization == 'linear_interpolation':
        points, densities = _linearly_interpolate_kde(
            labels,
            kde,
            distribution_samples,
            padding_factor
        )
    elif optimization == 'local_approximation':
        points, densities = _local_kde_optimization(labels, kde, k, atol)
    else:
        raise ValueError("'optimization' must be either 'linear_interpolation', 'local_approximation', or None")
    approx = (points, densities)

    densities = densities.reshape(densities.shape[0], -1)

    if return_optimization and approx is not None:
        return densities, approx
    else:
        return densities

def generate_weights(
        densities,
        density_mapping=None
    ):
    """
    Generates a list of weights, where the index of each weight corresponds to the density
    at the index of the provides list of density. The sum of all weights in the returned
    list of weights will be normalized to 1.

    Args:
        densities: A NumPy array of densities, arranged as a column vector
        density_mapping: Optional, default :code:`None`. A float to float function
            that converts density values to weights. When set to :code:`None`, densities
            are converted to weights be taking the reciprocal of each density value.
    Returns:
        A list of weights, normalized to 1.

    Example:

     .. code-block:: python

        >>> labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2])
        >>> kde = imbal.regression.fit_kde(labels, bin_count=3)
        >>> densities = imbal.regression.get_densities(labels, kde)
        >>> weights = imbal.regression.generate_weights(densities)

        >>> print(weights)
        [0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028,
        0.028, 0.028, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055, 0.163, 0.163]
    """
    if density_mapping is None:
        # Use uniform mapping for weights
        weights = 1 /np.array(densities)
    else:
        # Use function mapping for weights
        vectorized_function = np.vectorize(density_mapping)
        weights = vectorized_function(densities)

    weights = weights / np.sum(weights)

    return weights

def _local_kde_optimization(
        labels,
        kde,
        k,
        atol
):
    sort_indices = np.argsort(labels)
    sorted_labels = labels[sort_indices].reshape(-1,)
    inverse_sort = np.argsort(sort_indices)
    bandwidth = kde.bandwidth_
    inverse_gaussian = lambda x: sqrt(-2 * log(x * (bandwidth * sqrt(2 * pi)))) * bandwidth
    atol_by_n = atol / labels.shape[0]
    delta = inverse_gaussian(atol_by_n)


    sample_densities = []
    low_index = 0
    high_index = 0
    for label in sorted_labels:
        while sorted_labels[low_index] < label - delta:
            low_index += 1
        while high_index < sorted_labels.shape[0] and sorted_labels[high_index] < label + delta:
            high_index += 1

        if k is not None and high_index - low_index > k:
            stride = ceil((high_index - low_index) / k)
            samples = sorted_labels[low_index:high_index:stride]
        else:
            samples = sorted_labels[low_index:high_index]

        def mini_kde(value, data):
            u = (data - value) / bandwidth
            return np.sum(np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)) / (data.shape[0] * bandwidth)

        sample_densities.append(mini_kde(label, samples) * (high_index - low_index) / labels.shape[0])
    sample_densities = np.array(sample_densities).reshape(-1,)
    return sorted_labels[inverse_sort], sample_densities[inverse_sort]

def _linearly_interpolate_kde(
        labels,
        kde,
        distribution_samples,
        padding_factor
):
    labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    D = labels.shape[1]

    # Get min, max, step per dimension
    label_min, label_max, step = get_label_bin_bounds(labels, distribution_samples, padding_factor)
    # Create 1D linspace per dimension
    grids_1d = [np.linspace(lo, hi, distribution_samples + 1) for lo, hi in zip(label_min, label_max)]
    # Create full N-D meshgrid of sample points
    mesh = np.meshgrid(*grids_1d, indexing='ij')
    sample_points = np.stack(mesh, axis=-1).reshape(-1, D)  # (num_points, D)

    interpolate_densities = np.exp(kde.score_samples(sample_points))
    interpolate_densities = interpolate_densities.reshape([distribution_samples + 1] * D)
    interpolator = RegularGridInterpolator(tuple(grids_1d), interpolate_densities)
    sample_densities = interpolator(labels)
    return sample_points, sample_densities
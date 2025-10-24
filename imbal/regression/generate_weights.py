import numpy as np
from math import sqrt, pi, log, ceil

from sklearn.neighbors import KernelDensity

from imbal.util.sample_weighting import get_label_bin_bounds
from scipy.interpolate import RegularGridInterpolator

def get_densities(
        labels,
        bandwidth,
        interpolation_method=None,
        interpolation_samples=None,
        atol=0,
        padding_factor=0.01,
        return_interpolation_samples=False
):
    r"""
    This function exists as a means of extracting densities from a scitkit-learn
    :code:`KernelDensity` object. As a part of this function, some approaches to
    improve time efficiency for KDE sampling have been included, which can
    cost at the cost of a small error in the desnity values sampled.

    Kernel density estimatinon is defined as:

    .. math::

       \hat{f}_h(x)=\frac{1}{n}\sum_{i=1}^{n}K_h(x-x_i)

    By definition, to compute the KDE of a single point, we require a summation of the
    density kernel centered at each known data point. Therefore, computing the KDE
    for a single label is an :math:`O(n)` operation, and computing the KDE for each
    label is an :math:`O(n^2)` operation. This means retreiving the densities
    for a large dataset will take a long time. To combat this, :code:`get_densities`
    implements some estimation methods to reduce compute time.

    The first of these estimation methods is allowing some absolute tolerance for the
    error of each KDE value (See: :code:`atol`). For one dimensional data, this absolute tolerance
    is applied by first sorting the data, then starting from the smallest data point,
    tracking which points are within some delta of the current point, such that all points
    outside the delta range contribute an error of no more than :code:`atol/n`, where :code:`n`
    is the number of data points. By working with the points in ascending order, the delta bounds
    for each point can be found in :math:`O(1)` amortized time. We use this method as opposed to a binary
    search to find the bounds for each point, which would be an :math:`O(nlogn)` operation.

    For multidimensional data, we leverage :code:`scikit-learn`'s built-in atol for KDEs, which
    utilizes a :code:`KDTree` to find its tolerance bounds. This way, the gains in computational
    performance can still be utilized for one dimensional data, without sacrificing loss of generality.

    The second estimation method is an interpolation method. This method works by sampling
    evenly spaced points across the span of the provided dataset from the KDE, then using
    one of `scipy's RegularGridInterpolator interpolation methods <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html>`_
    to extrapolate the density values for the dataset. This method tends to be faster
    than using an absolute tolerance, at the cost of sometimes yielding less accurate results
    (with no guarentee of some maximum error).

    Args:
        bandwidth: The bandwidth that should be used to generate the KDE used for densit sampling
        labels: A NumPy array of labels, arranged as a column vector
        interpolation_method: Optional, default :code:`None`. When not set as :code:`None`, will
            be passed to a `scipy RegularGridInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html>`_
            as the method that should be used to interpolate between sampled points.
        interpolation_samples: Optional, default :code:`None`. The number of points to be sampled from the density distribution for interpolation.
            When set to :code:`None`, this value is computed as :code:`labels.shape[0] / 10`. If KDE was generated
            using :code:`imbal.regression.fit_kde`, :code:`steps_per_bin*bin_count` tends to be a good
            value for :code:`distribution_samples`. This parameter is ignored when :code:`interpolation_method` is set to :code:`None`.
        atol: Optional, default :code:`1e-4`. The maximum absolute error for each sample in the
            KDE.
        return_interpolation_samples: Optional, default :code:`False` If set to true, returns a tuple
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

    Examples:

     .. code-block:: python

        >>> labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2])
        >>> kde_bandwidth = imbal.regression.fit_kde(labels, bin_count=3)
        >>> densities = imbal.regression.get_densities(labels, kde_bandwidth)

        >>> print(densities)
        [0.549, 0.549, 0.549, 0.549, 0.549, 0.549, 0.549, 0.549, 0.549, 0.549,
         0.549, 0.549, 0.318, 0.318, 0.318, 0.318, 0.318, 0.318, 0.111, 0.111]

     .. code-block:: python

        >>> # Local Approximation Example
        >>> labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2])
        >>> kde_bandwidth = imbal.regression.fit_kde(labels, bin_count=3)
        >>> local_approx_densities = imbal.regression.get_densities(
        >>>     labels,
        >>>     kde_bandwidth,
        >>>     atol=0.2
        >>> )

        >>> # For this example dataset, there are no errors, even for higher tolerance values
        >>> print(local_approx_densities)
        [0.549, 0.549, 0.549, 0.549, 0.549, 0.549, 0.549, 0.549, 0.549, 0.549,
         0.549, 0.549, 0.318, 0.318, 0.318, 0.318, 0.318, 0.318, 0.111, 0.111]

     .. code-block:: python

        >>> # Linear Interpolation Example
        >>> labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2])
        >>> kde_bandwidth = imbal.regression.fit_kde(labels, bin_count=3)
        >>> linear_interpolation_densities = imbal.regression.get_densities(
        >>>     labels,
        >>>     kde_bandwidth,
        >>>     interpolation_method='linear',
        >>>     interpolation_samples=5
        >>> )

        >>> print(linear_interpolation_densities)
        [0.543, 0.543, 0.543, 0.543, 0.543, 0.543, 0.543, 0.543, 0.543, 0.543,
         0.543, 0.543, 0.314, 0.314, 0.314, 0.314, 0.314, 0.314, 0.112, 0.112]

    """
    if interpolation_samples is None:
        interpolation_samples = round(labels.shape[0] / 10)

    labels = labels.reshape(labels.shape[0], -1)

    kde = KernelDensity(bandwidth=bandwidth, atol=atol).fit(labels)

    # Use KDE estimation to generate weights
    points, densities = None, None
    if interpolation_method is None:
        if labels.shape[1] == 1:
            points, densities = _local_kde_estimation(labels, kde, atol=atol)
        else:
            densities = np.exp(kde.score_samples(labels).reshape(-1,))
    else:
        points, densities = _linearly_interpolate_kde(
            labels,
            kde,
            interpolation_samples,
            padding_factor,
            interpolation_method
        )

    approx = (points, densities)

    densities = densities.reshape(densities.shape[0], -1)

    if return_interpolation_samples and approx is not None:
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

def _local_kde_estimation(
        labels,
        kde,
        atol=0
):
    labels = labels.reshape(-1, )
    sort_indices = np.argsort(labels)
    sorted_labels = labels[sort_indices]
    inverse_sort = np.argsort(sort_indices)
    bandwidth = kde.bandwidth_
    inverse_gaussian = lambda x: sqrt(-2 * log(x * (bandwidth * sqrt(2 * pi)))) * bandwidth
    atol_by_n = atol / labels.shape[0]
    if atol == 0:
        delta = np.max(labels) - np.min(labels)
    else:
        delta = inverse_gaussian(atol_by_n)
    sample_densities = []
    low_index = 0
    high_index = 0
    for label in sorted_labels:
        while sorted_labels[low_index] < label - delta:
            low_index += 1
        while high_index < sorted_labels.shape[0] and sorted_labels[high_index] < label + delta:
            high_index += 1
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
        interpolation_samples,
        padding_factor,
        interpolation_method
):
    labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    D = labels.shape[1]

    # Get min, max, step per dimension
    label_min, label_max, step = get_label_bin_bounds(labels, interpolation_samples, padding_factor)
    # Create 1D linspace per dimension
    grids_1d = [np.linspace(lo, hi, interpolation_samples + 1) for lo, hi in zip(label_min, label_max)]
    # Create full N-D meshgrid of sample points
    mesh = np.meshgrid(*grids_1d, indexing='ij')
    sample_points = np.stack(mesh, axis=-1).reshape(-1, D)  # (num_points, D)

    interpolate_densities = np.exp(kde.score_samples(sample_points))
    interpolate_densities = interpolate_densities.reshape([interpolation_samples + 1] * D)
    interpolator = RegularGridInterpolator(tuple(grids_1d), interpolate_densities, method=interpolation_method)
    sample_densities = interpolator(labels)
    return sample_points, sample_densities
import numpy as np
from math import sqrt, pi, log

from sklearn.neighbors import KernelDensity

from imbal.util.backend.sample_weighting import get_label_bin_bounds
from scipy.interpolate import RegularGridInterpolator

import tensorflow as tf

def get_sample_densities(
    labels,
    bandwidth,
    distribution=None,
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
    implements some approximation methods to reduce compute time.

    The first of these approximation methods is allowing some absolute tolerance for the
    error of each KDE value (See: :code:`atol`). This method is motivated by the fact
    that points that are far from one another contribute very little to each of their
    respective KDEs. By ignoring points far away from each point in the KDE calculation,
    we can achieve a faster KDE calculation while ensuring a maximum possible error.

    For one dimensional data, we use a
    custom implementation that tends to yeilds results faster, which is implemented as follows:

    - We define :math:`\delta` such that :math:`K(\delta) = \text{atol}`, where :math:`K` is the
      Gaussian kernel. Or in other words, :math:`\delta` is equal to the distance from the center
      of the Gaussian kernel which will produce a density value of :code:`atol` (Concretely,
      :math:`\delta = K^{-1}(\text{atol})`). Points beyond this
      distance :math:`\delta` will produce a density less than :code:`atol`.
    - We sort the data, then iterate through the list of data points. From the current point, :math:`p`,
      we find all points :math:`x \in [p - \delta, p + \delta]`
    - For each point, we can determine a localized KDE estimate using only the points that fall within the
      delta range, that normalize the local KDE to the same scale as the full KDE.

    The result of this method is an apporixmation of the full KDE, such that each approximated point has an
    error no greater than :math:`\text{atol}/n`, therefore the total error is no more than :code:`atol`.

    By working with the points in ascending order, the delta bounds
    for each point can be found in :math:`O(1)` amortized time. We use this method as opposed to a binary
    search to find the bounds for each point, which would be an :math:`O(log(n))` operation per point.

    For multidimensional data, we leverage :code:`scikit-learn`'s built-in atol for KDEs, which
    utilizes a :code:`KDTree` to find its tolerance bounds. This way, the gains in computational
    performance can still be utilized for one dimensional data, without sacrificing loss of generality.

    The second approximation method is an interpolation method. This method works by sampling
    evenly spaced points across the span of the provided dataset from the KDE, then using
    one of `scipy's RegularGridInterpolator interpolation methods <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html>`_
    to extrapolate the density values for the dataset. This method tends to be faster
    than using an absolute tolerance, at the cost of sometimes yielding less accurate results
    (with no guarentee of some maximum error).

    Args:
        labels: A NumPy array of labels, arranged as a column vector
        bandwidth: The bandwidth that should be used to generate the KDE used for densit sampling
        distribution: Optional, default :code:`None`. A list of labels used to generate the KDE distribution. If set
            to :code:`None`, the provided :code:`labels` will be used to generate the distribution.
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
            the data range used for binning for the histogram. See :doc:`imbal.regression.fit_kde </imbal/regression/fit_kde>`.

    Returns:
        A NumPy array of densities, arranged as a column vector

    Examples:

     .. code-block:: python

        >>> labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 5])
        >>> kde_bandwidth = imbal.regression.fit_kde(labels, bin_count=3)
        >>> densities = imbal.regression.get_densities(labels, kde_bandwidth)

        >>> print(densities)
        [[1.0930867], [1.0930867], [1.0930867], [1.0930867], [1.0930867],
         [1.0930867], [0.5465676], [0.5465676], [0.5465676], [0.1821784]]

     .. code-block:: python

        >>> # Local Approximation Example
        >>> labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 5])
        >>> kde_bandwidth = imbal.regression.fit_kde(labels, bin_count=3)
        >>> local_approx_densities = imbal.regression.get_densities(
        >>>     labels,
        >>>     kde_bandwidth,
        >>>     atol=1e-3
        >>> )

        >>> # For this example dataset, small errors are found in some densities
        >>> print(local_approx_densities)
        [[1.0930705], [1.0930705], [1.0930705], [1.0930705], [1.0930705],
         [1.0930705], [0.5465353], [0.5465353], [0.5465353], [0.1821784]]

     .. code-block:: python

        >>> # Linear Interpolation Example
        >>> labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 5])
        >>> kde_bandwidth = imbal.regression.fit_kde(labels, bin_count=3)
        >>> linear_interpolation_densities = imbal.regression.get_densities(
        >>>     labels,
        >>>     kde_bandwidth,
        >>>     interpolation_method='linear',
        >>>     interpolation_samples=5
        >>> )

        >>> # For this example dataset, small errors are found in all densities
        >>> print(linear_interpolation_densities)
        [[1.0392918], [1.0392918], [1.0392918], [1.0392918], [1.0392918],
         [1.0392918], [0.5255651], [0.5255651], [0.5255651], [0.1687906]]

    """
    if interpolation_samples is None:
        interpolation_samples = round(labels.shape[0] / 10)

    input_label_shape = labels.shape
    labels = labels.reshape(labels.shape[0], -1)

    fit_data = distribution if distribution is not None else labels
    fit_data = fit_data.reshape(fit_data.shape[0], -1)

    kde = KernelDensity(bandwidth=bandwidth, atol=atol).fit(fit_data)

    # Use KDE approximation to generate weights
    points, densities = None, None
    if interpolation_method is None:
        if labels.shape[1] == 1:
            points, densities = _local_kde_approximation(labels, kde, atol=atol)
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

    densities = densities.reshape(input_label_shape)

    if return_interpolation_samples and approx is not None:
        return densities, approx
    else:
        return densities

def generate_sample_weights(
        densities,
        density_mapping=None
    ):
    """
    Generates a list of weights, where the index of each weight corresponds to the density
    at the index of the provides list of density. The sum of all weights in the returned
    list of weights will be normalized to :math:`n`.

    Normally, it is standard to normalize weights to :math:`1`. However, when no weights
    are provided to Tensorflow, its default behavior is to assign a weight of :math:`1`
    to each sample, meaning the total weight for the dataset is :math:`n`. Straying from
    this pattern would affect the scale of calculated loss values, which would also
    have an impact on how learning rates perform, therefore we have decided to align
    our weight generation implementations as closely as possible with Tensorflow's
    default behavaiors.

    Args:
        densities: A NumPy array of densities, arranged as a column vector
        density_mapping: Optional, default :code:`None`. A float to float function
            that converts density values to weights. When set to :code:`None`, densities
            are converted to weights be taking the reciprocal of each density value.
    Returns:
        A list of weights, normalized to :math:`n`.

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
        weights = 1 / np.array(densities)
    else:
        # Use function mapping for weights
        vectorized_function = np.vectorize(density_mapping)
        weights = vectorized_function(densities)

    weights = weights / np.sum(weights) * weights.shape[0]

    return weights

def _local_kde_approximation(
        labels,
        kde,
        atol=0
):

    unsigned_conversion = False
    # Ensure data is of proper type (labels must be signed data)
    if labels.dtype == np.uint8 or labels.dtype == np.ubyte:
        labels = labels.astype(np.int8)
        unsigned_conversion = True
    elif labels.dtype == np.uint16 or labels.dtype == np.ushort:
        labels = labels.astype(np.int16)
        unsigned_conversion = True
    elif labels.dtype == np.uint32 or labels.dtype == np.uintc:
        labels = labels.astype(np.int32)
        unsigned_conversion = True
    elif labels.dtype == np.uint64 or labels.dtype == np.ulong:
        labels = labels.astype(np.int64)
        unsigned_conversion = True
    elif labels.dtype == tf.uint8:
        labels = labels.astype(tf.int8)
        unsigned_conversion = True

    # Example of where this fails
    # Regression problem with uint8 values: [0, 20, 200, 185, 40]
    # Converting to int8 values yields: [0, 20, -56, -71, 40]
    if unsigned_conversion and min(labels) < 0:
        raise ValueError("When getting sample densities using imbal's custom KDE approximation, data should be passed as a signed type."
                      " Failed to successfully cast from an unsigned type to a signed type (negative values appeared).")

    labels = labels.reshape(-1, ).astype(np.float32)
    sort_indices = np.argsort(labels)
    sorted_labels = labels[sort_indices]
    inverse_sort = np.argsort(sort_indices)
    bandwidth = kde.bandwidth_
    inverse_gaussian = lambda x: sqrt(-2 * log(x * (bandwidth * sqrt(2 * pi)))) * bandwidth
    if atol == 0:
        delta = np.max(labels) - np.min(labels)
    else:
        delta = inverse_gaussian(atol)
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
            return np.sum(np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)) / bandwidth

        sample_densities.append(mini_kde(label, samples) / labels.shape[0])
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
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from imbal.util.sample_weighting import calculate_bin_count, get_label_bin_bounds
import itertools

def fit_kde(
        labels,
        bandwidth='kl_divergence',
        average_samples_per_bin=100,
        bin_count=None,
        steps_per_bin = 10,
        fine_search = 10,
        tolerance = 1e-3,
        padding_factor=0.01
):
    """
    Automatically determine a bandwidth and gaussian KDE curve best fits
    the labels provided.

    Args:
        labels: A NumPy array of labels, arranged as a column vector
        bandwidth: Optional, default 'kl_divergence'. Can be a number equal to
            the desired KDE bandwidth, or a string indicating the method to use to
            determine bandwidth. If set to :code:`scott` or :code:`silverman`, the
            bandwidth will be determined by using either Scott's or Silverman's
            rule-of-thumb. If set to :code:`kl_divergence`, or :code:`ratio`,
            an iterative approach will be used to determine the best bandwidth to minimize
            the specified heuristic. If :code:`kl_divergence`, will try to minimize the KL
            divergence between the KDE and the normalized histogram (area under the
            histogram is 1). If :code:`ratio`, will try to minimize the difference between
            the ratio of the highest frequency bin count to the lowest frequency bin count in the
            histogram, and the ratio between the average KDE densities in the parts of the KDE curve contained
            within those bins.
        average_samples_per_bin: Optional, default :code:`100`. Determines the
            number of bins used for histogram-based KDE approximation by the number of datapoints. For
            example, a dataset with 14500 data points with :code:`average_samples_per_bin` set to :code:`100`
            will have 145 bins.
        bin_count: Optional, default :code:`None`. The number of bins that
            should be used for the histogram-based KDE approximation.
            If set, overrides :code:`average_samples_per_bin`.
        steps_per_bin: Optional, default :code:`10`. Determines the number of
            steps per bin that should be used for KDE optimizations.
        fine_search: Optional, default :code:`10`. For iterative approaches only. Determines
            the number of checks to perform on each step of the iteration. A higher value will
            take longer, but is more likely to yield accurate results.
        tolerance: Optional, default :code:`1e-3`. For iterative approaches only. Determines
            the allowed maximum heuristic value before stopping in instances where the heuristic
            approaches 0. Prevents infinite iteration approaching 0.
        padding_factor: Optional, default :code:`0.01`. Used to add a small padding to
            the data range used for binning for the histogram. There are some instances where many datapoints in
            a dataset fall on the maximum or minimum. When viewed visually, the peak of the found KDE curve may
            appear to be on the edge, or slightly outside, of its corresponding bin (due to limited
            pixel resolution when plotting), which is undesirable for visual comparison. By padding, we can slightly increase
            the width of the histogram bins, shifting their bounds and allowing these peaks to appear
            inside the bins instead.

    Returns:
        A scikit-learn KernelDenity object.

    Example:

    .. code-block:: python

        >>> # For the sake of this example, assume a dataset has already been stored in the variable 'data'

        >>> kde = imbal.regression.fit_kde(data, bin_count=10)
        >>> log_densities = kde.score_samples(data)

        >>> print(log_densities)
        [-2.948, -3.961, -4.997, -2.948, -3.605, -2.885,
         -4.244, -4.148, -3.989, -3.961, -2.885, -3.078,
         -4.275, -2.885, -3.961, -3.989, -5.822, -2.800...

    """
    bin_count = calculate_bin_count(labels, bin_count, average_samples_per_bin)

    found_bandwidth = bandwidth
    if bandwidth in ['ratio', 'kl_divergence']:
        # Use iterative, "binned-based" approach to approximate KDE
        found_bandwidth = _iterative_kde_approximation(
            labels,
            bin_count=bin_count,
            bandwidth=bandwidth,
            steps_per_bin=steps_per_bin,
            fine_search=fine_search,
            tolerance=tolerance,
            padding_factor=padding_factor,
        )

    # Use literal or explicit bandwidth to approximate KDE
    # kde = KernelDensity(bandwidth=found_bandwidth, atol=atol)
    # kde.fit(labels.reshape(labels.shape[0], -1))

    return found_bandwidth

def plot_kde_1d(
        labels,
        kde,
        average_samples_per_bin=100,
        bin_count=None,
        padding_factor=0.01,
        approximation=None,
        use_axes=None,
        save_figure=None
) -> None:
    """

    Args:
        labels: A NumPy array of labels, arranged as a column vector
        kde: A scikit-learn KernelDensity object.
        average_samples_per_bin: Optional, default :code:`100`. Determines the
            number of bins used for histogram-based KDE approximation by the number of datapoints. For
            example, a dataset with 14500 data points with :code:`average_samples_per_bin` set to :code:`100`
            will have 145 bins.
        bin_count: Optional, default :code:`None`. The number of bins that
            should be used for the histogram-based KDE approximation.
            If set, overrides :code:`average_samples_per_bin`.
        padding_factor: Optional, default :code:`0.01`. Used to add a small padding to
            the data range used for binning for the histogram. There are some instances where many datapoints in
            a dataset fall on the maximum or minimum. When viewed visually, the peak of the found KDE curve may
            appear to be on the edge, or slightly outside, of its corresponding bin (due to limited
            pixel resolution when plotting), which is undesirable for visual comparison. By padding, we can slightly increase
            the width of the histogram bins, shifting their bounds and allowing these peaks to appear
            inside the bins instead.
        approximation: Optional, default :code:`None`. A tuple containing a list of x and y
            pairs, which will be plotted over the KDE curve. Used to show how well approximations
            of KDE perform.
        use_axes: Optional, default :code:`None`. If a matplotlib :code:`Axes` object is
            passed, the figure will be written to the :code:`Axes` object instead of
            being immediately displayed.
        save_figure: Optional, default :code:`None`. The path where the figure should
            be saved to on the local system, as a string. Only saves is not set to :code:`None`.

    Returns:
        :code:`None`

    Example:

    .. code-block:: python

        >>> # For the sake of this example, assume a dataset has already been stored in the variable 'data'

        >>> kde = imbal.regression.fit_kde(data, bin_count=10)
        >>> imbal.regression.plot_kde(data, kde, bin_count=10, save_figure='plot.png')

    Below is the resultant graph saved to :code:`plot.png`:

    .. figure:: images/example_kde_plot.png
       :scale: 85 %
       :alt: A histogram plot of the data from the example above
    """
    bin_count = calculate_bin_count(labels, bin_count, average_samples_per_bin)
    labels = np.sort(labels.reshape(-1, ))
    high_freq_bin_count, high_freq_bin_index, low_freq_bin_count, low_freq_bin_index = _determine_high_low_freq_bins(labels,
                                                                                                         bin_count,
                                                                                                         padding_factor)
    label_min, label_max, step = get_label_bin_bounds(labels, bin_count, padding_factor)
    label_min = label_min[0]
    label_max = label_max[0]
    step = step[0]
    low_freq_bin_index = low_freq_bin_index[0]
    high_freq_bin_index = high_freq_bin_index[0]
    min_count = low_freq_bin_count
    max_count = high_freq_bin_count

    x_plot = np.linspace(label_min - 1, label_max + 1, 1000).reshape(-1, 1)
    log_dens = kde.score_samples(x_plot)
    if use_axes is not None:
        ax = use_axes
    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    ax.plot(x_plot, np.exp(log_dens), label='KDE Curve')
    if approximation is not None:
        plt.plot(approximation[0], approximation[1], label='Approximation', color='orange')
    ax.hist(labels, bins=[label_min + i * step for i in range(bin_count+1)], density=True, alpha=0.6, label='Histogram')
    f_min_bar = label_min + (low_freq_bin_index + .5) * step
    ax.axvline(x=f_min_bar, color='red', linestyle='--', linewidth=2)
    ax.text(f_min_bar, ax.get_ylim()[1] * 0.02, f'f_min = {min_count}',
             rotation=90, color='red',
             verticalalignment='bottom', horizontalalignment='right')

    f_max_bar = label_min + (high_freq_bin_index + .5) * step
    ax.axvline(x=f_max_bar, color='red', linestyle='--', linewidth=2)
    ax.text(f_max_bar, ax.get_ylim()[1] * 0.02, f'f_max = {max_count}',
             rotation=90, color='red',
             verticalalignment='bottom', horizontalalignment='right')

    ax.set_title(
        f'KDE (f_max/f_min = {max_count / min_count:.1f}, bandwidth = {kde.bandwidth_:.3f}, bins = {bin_count})')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)

    if save_figure is not None:
        plt.savefig(save_figure)
    if use_axes is None:
        plt.show()


def _iterative_kde_approximation(
    labels,
    bin_count=10,
    bandwidth=None,
    steps_per_bin=10,
    fine_search = 10,
    tolerance = 1e-3,
    padding_factor=0.01
) -> KernelDensity | tuple:

    # Get bounds and bin width
    label_min, label_max, step = get_label_bin_bounds(labels, bin_count, padding_factor)
    # Get bins with the highest frequency and lowest frequency
    high_freq_bin_count, high_freq_bin_index, low_freq_bin_count, low_freq_bin_index = _determine_high_low_freq_bins(labels, bin_count, padding_factor)

    spaced_even_curve = _evenly_sample_space(label_min, step, bin_count, steps_per_bin)
    histogram_values = _compute_histogram_areas(labels, bin_count, padding_factor, steps_per_bin)


    # Starting test ranges for bandwidth should be between 0.01*stddev and 3*stddev
    starting_bandwidth = float(np.std(labels)) * 1.49
    coarse_search = starting_bandwidth

    # Track best results during iteration
    best_heuristic = None
    best_bandwidth = starting_bandwidth

    def kl_divergence_heuristic(kde_curve) -> float:
        filtered_histogram = histogram_values
        filtered_curve = spaced_even_curve
        data_shape = filtered_curve.shape
        kde_scores = np.array([
            np.exp(kde_curve.score_samples(group))
            for group in filtered_curve.reshape(np.prod(data_shape))
        ])
        summed_kde = np.sum(kde_scores, axis=1)
        total_kde = np.sum(summed_kde, axis=0)
        normalized_kde = summed_kde / total_kde
        normalized_kde = normalized_kde.reshape(data_shape)
        return np.sum(filtered_histogram*np.log(filtered_histogram/(normalized_kde+1e-6)+ 1e-6))

    # Generate lists used for even-spaced sampling across lowest and highest frequency bins
    spaced_high_freq_bin = _evenly_sample_bin(label_min, step, low_freq_bin_index, steps_per_bin)

    spaced_low_freq_bin = _evenly_sample_bin(label_min, step, high_freq_bin_index, steps_per_bin)
    def ratio_heuristic(kde_curve) -> float:
        high_densities = np.exp(kde_curve.score_samples(spaced_high_freq_bin))
        low_densities = np.exp(kde_curve.score_samples(spaced_low_freq_bin))
        max_area = np.sum(high_densities) * step / steps_per_bin
        min_area = np.sum(low_densities) * step / steps_per_bin
        desired_ratio = high_freq_bin_count / low_freq_bin_count
        return abs(max_area / min_area - desired_ratio)

    heuristic_function = kl_divergence_heuristic
    if bandwidth == 'ratio':
        heuristic_function = ratio_heuristic
    labels = labels.reshape(labels.shape[0], -1)

    # Ensure at least one loop, and loop until ratio is within tolerance,
    # or search becomes too fine grain (perhaps ideal ratio is impossible)
    while best_heuristic is None or (best_heuristic > tolerance and coarse_search / fine_search > 1e-6):
        search_steps = round(fine_search)
        heuristic_contender = None
        bandwidth_contender = None

        for i in range(search_steps):
            current_bandwidth = best_bandwidth + (i - (search_steps-1)/2) * 2*coarse_search / fine_search
            # Prevent negative and zero bandwidths
            if current_bandwidth <= 1e-6:
                continue

            # Fit KDE with current bandwidth
            kde = KernelDensity(bandwidth=current_bandwidth)
            kde.fit(labels)

            heuristic = heuristic_function(kde)

            if bandwidth_contender is None or heuristic < heuristic_contender:
                heuristic_contender = heuristic
                bandwidth_contender = current_bandwidth


        if best_heuristic is None or heuristic_contender <= best_heuristic:
            # If the best contender in this search loop produces a
            # better density ratio than the previous best, update best
            best_heuristic = heuristic_contender
            best_bandwidth = bandwidth_contender
            coarse_search = coarse_search/fine_search
        else:
            # Otherwise, current search window provided no better
            # bandwidth values, break from search loop
            break

    return best_bandwidth


def _determine_high_low_freq_bins(labels, bin_count, padding_factor) -> tuple:

    labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    D = labels.shape[1]

    label_min, label_max, step = get_label_bin_bounds(labels, bin_count, padding_factor)
    bin_indices = np.floor((labels - label_min) / step).astype(int)
    bin_indices = np.clip(bin_indices, 0, bin_count - 1)

    # Count frequencies in each bin
    frequencies = np.zeros([bin_count] * D, dtype=int)
    for idx in bin_indices:
        frequencies[tuple(idx)] += 1

    high_freq_bin_index = np.unravel_index(np.argmax(frequencies), frequencies.shape)
    high_freq_count = frequencies[high_freq_bin_index]

    nonzero_mask = frequencies > 0
    if np.any(nonzero_mask):
        low_freq_bin_index = np.unravel_index(np.argmin(frequencies[nonzero_mask]), frequencies.shape)
        low_freq_bin_count = frequencies[low_freq_bin_index]
    else:
        low_freq_bin_index = None
        low_freq_bin_count = 0

    return high_freq_count, high_freq_bin_index, low_freq_bin_count, low_freq_bin_index

def _get_bin_bounds(label_min, step, bin_idx):
    """
    Given the min, step, and a bin index (tuple), return the coordinate range for that bin.
    """
    lower = label_min + step * np.array(bin_idx)
    upper = lower + step
    return lower, upper

def _evenly_sample_space(label_min, step, bin_count, steps_per_bin):
    dim_ranges = [range(bin_count) for _ in label_min]

    all_points = np.empty([bin_count for _ in label_min], dtype=object)
    for bin_idx in itertools.product(*dim_ranges):
        points = _evenly_sample_bin(label_min, step, np.array(bin_idx), steps_per_bin)
        all_points[bin_idx] = points
    return all_points

def _evenly_sample_bin(label_min, step, bin_idx, steps_per_bin):
    lower, upper = _get_bin_bounds(label_min, step, bin_idx)
    # Create 1D linspaces per dimension
    grids = [np.linspace(l, u, steps_per_bin + 1)[:-1] + step/2 for l, u in zip(lower, upper)]
    # Create full N-D sampling grid
    mesh = np.stack(np.meshgrid(*grids, indexing='ij'), axis=-1)
    return mesh.reshape(-1, len(step))  # Flatten to list of (N_points, D)

def _compute_histogram_areas(labels, bin_count, padding_factor, steps_per_bin):
    labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    D = labels.shape[1]

    label_min, label_max, step = get_label_bin_bounds(labels, bin_count, padding_factor)

    # Compute which bin each label falls into
    bin_indices = np.floor((labels - label_min) / step).astype(int)
    bin_indices = np.clip(bin_indices, 0, bin_count - 1)

    # Make an N-dimensional frequency grid
    freq = np.zeros([bin_count] * D, dtype=int)
    for idx in bin_indices:
        freq[tuple(idx)] += 1
    # Compute normalized histogram values per bin
    histogram_values = freq / np.prod(step) / labels.shape[0]

    return histogram_values
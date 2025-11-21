import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

from imbal.util.backend.sample_weighting import calculate_bin_count, get_label_bin_bounds
import itertools
from math import floor

def fit_kde(
        labels,
        fit_method='kl_divergence',
        average_samples_per_bin=100,
        bin_count=None,
        steps_per_bin = 10,
        num_candidates = 10,
        tolerance = 0,
        padding_factor=0.01
):
    """
    Determine the bandwidth which generates a KDE curve that best fits
    the labels provided, based on some rule of thumb or heuristic.

    For the best results, we have implemented an iterative fit function that aims to
    minimize the KL divergence between the area of the density-normalized histogram
    bins of the data (area of all bins sums to :math:`1`), and the area under the curve
    of the KDE at each bin. The fit is performed as follows:

    - Start by assuming that the bandwidth that best fits the data is
      between :math:`0.01` and :math:`3` times the standard deviation of the data.
    - We perform a beam search within this data range, searching :math:`k` canditates
      within the current data range.
    - For each bandwidth candidate, we compute the KL divergence between the area of each
      bin in the density-normalized histogram of the data and the area under the curve
      of the KDE within the corresponding data range for each bin.
    - After checking all :math:`k` candidates, we take the candidate with the lowest KL
      divergence, and perform a new round of beam searches, centered on the best candidate,
      and spanning the range of the neighboring candiates (ex. With :math:`k=10`, if a
      beam search from :math:`1` to :math:`10` found :math:`3` to be the best bandwidth
      candidate, the next round would be a beam search from :math:`2` to :math:`4`).

    There are two stopping criteria for this iterative fit method:

    - The candidates in the current round perform no better than the best candidate from the previous round.
    - The KL divergence is within some tolerance of zero (this method is disabled by default).

    In general, the KL divergence heuristic will reach a minimum greater than :math:`0` rather
    than find a true "perfect fit", so the second stopping criteria is disabled by default. Still
    though, in the scenarios where the KL divergence does approach :math:`0`, it can be nice
    to have some tolerance, as setting one can allow the iterative method to stop a few rounds
    earlier than what the first stopping criteria would allow.

    It is important to note that since the KL divergence
    is calculated by performing per-bin comparisons with a histogram of the data, the bandwidth
    fit found by this method is dependent on the number of bins that the data is divided into.
    In general, lower bin counts will result in a smoother KDE, and higher bin counts will
    result in a bumpy KDE. We calculate the AUC for each bin by
    performing midpoint sums within the bounds of the bin by sampling the KDE.

    The :code:`'scott'` and :code:`'silverman'` bandwidth fitting methods are explicit, "rule of thumb" methods
    for finding the bandwidth that take :math:`O(n)` time. The :code:`'kl_divergence'`
    method is an iterative method that takes :math:`O(rkn)` time, where :math:`k` is
    the number of searches performed per round (default :math:`10`), and :math:`r` is
    the numer of rounds it takes to reach a final value (determined by the stopping criteria described above. Based
    on our experiments, :math:`r` is typically between :math:`5` and :math:`10`).

    Args:
        labels: A NumPy array of labels, arranged as a column vector
        fit_method: Optional, default :code:`'kl_divergence'`. A string indicating the method to use to
            determine bandwidth. If set to :code:`'scott'` or :code:`'silverman'`, the
            bandwidth will be determined by using either Scott's or Silverman's
            rule-of-thumb. If set to :code:`'kl_divergence'`,
            an iterative approach will be used to determine the best bandwidth to minimize
            the specified heuristic. If :code:`'kl_divergence'`, will try to minimize the KL
            divergence between the KDE and the normalized histogram (area under the
            histogram is 1).
        average_samples_per_bin: Optional, default :code:`100`. Determines the
            number of bins used for histogram-based KDE approximation by the number of datapoints. For
            example, a dataset with 14500 data points with :code:`average_samples_per_bin` set to :code:`100`
            will have 145 bins.
        bin_count: Optional, default :code:`None`. The number of bins that
            should be used for the histogram-based KDE approximation.
            If set, overrides :code:`average_samples_per_bin`.
        steps_per_bin: Optional, default :code:`10`. Determines the number of
            steps per bin that should be used for KDE optimizations.
        num_candidates: Optional, default :code:`10`. For iterative approach only. Determines
            the number of candidates to check during each round of beam search. A higher value will
            take longer, but is more likely to yield accurate results.
        tolerance: Optional, default :code:`0`. For iterative approach only. Determines the tolerance
            within which the iterative heuristic can be considered close to 0, allowing iteration to end.
            Prevents infinite iteration approaching 0.
        padding_factor: Optional, default :code:`0.01`. Used to add a small padding to
            the data range used for binning for the histogram. There are some instances where many datapoints in
            a dataset fall on the maximum or minimum. When viewed visually, the peak of the found KDE curve may
            appear to be on the edge, or slightly outside, of its corresponding bin (due to limited
            pixel resolution when plotting), which is undesirable for visual comparison. By padding, we can slightly increase
            the width of the histogram bins, shifting their bounds and allowing these peaks to appear
            inside the bins instead.

    Returns:
        The found bandwidth that fits the distribution of the provided data, as a float.

    Example:

    .. code-block:: python

        >>> # For the sake of this example, assume a dataset has already been stored in the variable 'data'

        >>> fitted_bandwidth = imbal.regression.fit_kde(data, bin_count=10, tolerance=1e-3)
        >>> kde = KernelDensity(bandwidth=fitted_bandwidth)
        >>> kde.fit(data)
        >>> log_densities = kde.score_samples(data)

        >>> print(log_densities)
        [0.702, 0.634, -2.445, 0.535, 0.491, 0.154,
         0.287, 0.710, 0.059, -0.365, 0.691, -0.758,
         0.687, -0.336, 0.635, 0.594, 0.658, 0.697 ...



    """
    bin_count = calculate_bin_count(labels, bin_count, average_samples_per_bin)

    found_bandwidth = None
    if fit_method in ['kl_divergence']:
        # Use iterative, "binned-based" approach to approximate KDE
        found_bandwidth = _iterative_kde_approximation(
            labels,
            bin_count=bin_count,
            steps_per_bin=steps_per_bin,
            fine_search=num_candidates,
            tolerance=tolerance,
            padding_factor=padding_factor,
        )
    elif fit_method in ['scott', 'silverman']:
        labels = labels.reshape(labels.shape[0], -1)
        kde = KernelDensity(bandwidth=fit_method)
        kde.fit(labels)
        found_bandwidth = kde.bandwidth_

    assert found_bandwidth is not None

    return found_bandwidth

def plot_kde_1d(
        labels,
        bandwidth,
        average_samples_per_bin=100,
        bin_count=None,
        padding_factor=0.01,
        approximation=None,
        use_axes=None,
        save_figure=None,
        show_bandwidth=True,
        show_bin_count=True,
        show_extreme_frequencies=False
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
            the data range used for binning for the histogram. See :doc:`imbal.regression.fit_kde </imbal/regression/fit_kde>`.
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

        >>> found_bandwidth = imbal.regression.fit_kde(data, bin_count=10)
        >>> imbal.regression.plot_kde_1d(
        >>>     data,
        >>>     found_bandwidth,
        >>>     bin_count=10,
        >>>     save_figure='plot.png'
        >>> )

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

    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(labels.reshape(-1, 1))

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

    if show_extreme_frequencies:
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

    title_strings = []
    if show_bandwidth:
        title_strings.append(f'bandwidth = {bandwidth:.3f}')
    if show_bin_count:
        title_strings.append(f'bin_count = {bin_count}')

    title_details = ''
    if len(title_strings) > 0:
        title_details = f' ({", ".join(title_strings)})'

    ax.set_title(f'KDE{title_details}')
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
    steps_per_bin=10,
    fine_search = 10,
    tolerance = 0,
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

    heuristic_function = kl_divergence_heuristic
    labels = labels.reshape(labels.shape[0], -1)

    search_min = 0
    search_max = starting_bandwidth * 2

    # Ensure at least one loop, and loop until ratio is within tolerance,
    # or search becomes too fine grain (perhaps ideal ratio is impossible)
    while best_heuristic is None or (best_heuristic > tolerance and (search_max - search_min) > 1e-9):
        search_steps = round(fine_search)
        heuristic_contender = None
        bandwidth_contender = None

        for i in range(search_steps):
            if search_steps % 2 == 1 and i == floor(search_steps / 2):
                continue

            current_bandwidth = (search_max - search_min) / search_steps * (i + 0.5)
            # Prevent negative and zero bandwidths
            if current_bandwidth <= 1e-9:
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
            search_min = best_bandwidth - (search_max - search_min) / search_steps
            search_max = best_bandwidth + (search_max - search_min) / search_steps
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
    if not np.any(nonzero_mask):
        return high_freq_bin_index, high_freq_count, None, 0

    nonzero_indices = np.argwhere(nonzero_mask)
    nonzero_counts = frequencies[nonzero_mask]
    min_freq = np.min(nonzero_counts)

    low_freq_candidates = nonzero_indices[nonzero_counts == min_freq]

    distances = np.linalg.norm(low_freq_candidates - np.array(high_freq_bin_index), axis=1)

    low_freq_bin_index = tuple(low_freq_candidates[np.argmax(distances)])
    low_freq_bin_count = frequencies[low_freq_bin_index]

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
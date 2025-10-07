import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from imbal.util.sample_weighting import calculate_bin_count, get_label_bin_bounds

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
            rule-of-thumb. If set to :code:`kl_divergence`, :code:`mse`, or :code:`ratio`,
            an iterative approach will be used to determine the best bandwidth to minimize
            the specified heuristic. If :code:`kl_divergence`, will try to minimize the KL
            divergence between the KDE and the normalized histogram (area under the
            histogram is 1). If :code:`mse`, will try to minimize the MSE between the KDE and the
            normalized histogram. If :code:`ratio`, will try to minimize the difference between
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
            the data range used for binning. This padding should be specified as a percentage
            of the range of the data labels. This padding allows for a more graceful
            handling of scenarios where the minimum or maximum of the labels is the most
            frequent value in the labels.

    Returns:
        A scikit-learn KernelDensity object.

    """
    bin_count = calculate_bin_count(labels, bin_count, average_samples_per_bin)

    if bandwidth in ['mse', 'ratio', 'kl_divergence']:
        # Use iterative, "binned-based" approach to approximate KDE
        kde = _iterative_kde_approximation(
            labels,
            bin_count=bin_count,
            bandwidth=bandwidth,
            steps_per_bin=steps_per_bin,
            fine_search=fine_search,
            tolerance=tolerance,
            padding_factor=padding_factor,
        )
    else:
        # Use literal or explicit bandwidth to approximate KDE
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(labels.reshape(-1, 1))

    return kde

def plot_kde(
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
            the data range used for binning. This padding should be specified as a percentage
            of the range of the data labels. This padding allows for a more graceful
            handling of scenarios where the minimum or maximum of the labels is the most
            frequent value in the labels.
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
    """
    bin_count = calculate_bin_count(labels, bin_count, average_samples_per_bin)

    labels = np.sort(labels.reshape(-1, ))
    high_freq_bin, high_freq_bin_index, low_freq_bin, low_freq_bin_index = _determine_high_low_freq_bins(labels,
                                                                                                         bin_count,
                                                                                                         padding_factor)
    label_min, label_max, step = get_label_bin_bounds(labels, bin_count, padding_factor)
    min_count = low_freq_bin.shape[0]
    max_count = high_freq_bin.shape[0]

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

    ax.hist(labels, bins=[label_min + i * step for i in range(bin_count)], density=True, alpha=0.6, label='Histogram')
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

    # Data formatting
    labels = np.sort(labels.reshape(-1,))

    # Get bounds and bin width
    label_min, label_max, step = get_label_bin_bounds(labels, bin_count, padding_factor)
    # Get bins with the highest frequency and lowest frequency
    high_freq_bin, high_freq_bin_index, low_freq_bin, low_freq_bin_index = _determine_high_low_freq_bins(labels, bin_count, padding_factor)
    desired_ratio = high_freq_bin.shape[0] / low_freq_bin.shape[0]

    # Generate lists used for even-spaced sampling across lowest and highest frequency bins
    spaced_high_freq_bin = np.linspace(high_freq_bin[0], high_freq_bin[0] + step, steps_per_bin).reshape(-1, 1)
    spaced_low_freq_bin = np.linspace(low_freq_bin[0], low_freq_bin[0] + step, steps_per_bin).reshape(-1, 1)

    spaced_even_curve = np.linspace(label_min, label_max, steps_per_bin*bin_count).reshape(-1, 1)
    bin_counts = [labels[(labels < label_min + step * (i + 1)) & (labels >= label_min + step * i)].shape[0] for i in range(bin_count)]
    even_counts = np.repeat(bin_counts, steps_per_bin)
    histogram_values = even_counts / step / labels.shape[0]


    # Starting test ranges for bandwidth should be between 0.01*stddev and 3*stddev
    starting_bandwidth = float(np.std(labels)) * 1.49
    coarse_search = starting_bandwidth

    # Track best results during iteration
    best_kde = None
    best_heuristic = None
    best_bandwidth = starting_bandwidth

    def kl_divergence_heuristic(kde_curve) -> float:
        filtered_histogram = histogram_values[histogram_values > 0]
        filtered_curve = spaced_even_curve[histogram_values > 0]
        kde_densities = np.exp(kde_curve.score_samples(filtered_curve))
        return np.sum(filtered_histogram*np.log(filtered_histogram/kde_densities))

    def mse_heuristic(kde_curve) -> float:
        kde_densities = np.exp(kde_curve.score_samples(spaced_even_curve))
        return np.mean(np.square(histogram_values - kde_densities))

    def ratio_heuristic(kde_curve) -> float:
        high_densities = np.exp(kde_curve.score_samples(spaced_high_freq_bin))
        low_densities = np.exp(kde_curve.score_samples(spaced_low_freq_bin))
        max_area = (-high_densities[0] / 2 - high_densities[-1] / 2 + np.sum(high_densities)) * step / steps_per_bin
        min_area = (-low_densities[0] / 2 - low_densities[-1] / 2 + np.sum(low_densities)) * step / steps_per_bin
        return abs(max_area / min_area - desired_ratio)

    heuristic_function = kl_divergence_heuristic
    if bandwidth == 'mse' :
        heuristic_function = mse_heuristic
    elif bandwidth == 'ratio':
        heuristic_function = ratio_heuristic
    labels = labels.reshape(-1, 1)

    # Ensure at least one loop, and loop until ratio is within tolerance,
    # or search becomes too fine grain (perhaps ideal ratio is impossible)
    while best_kde is None or (best_heuristic > tolerance and coarse_search / fine_search > 1e-6):
        search_steps = round(fine_search)
        kde_contender = None
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
                kde_contender = kde
                heuristic_contender = heuristic
                bandwidth_contender = current_bandwidth


        if best_heuristic is None or heuristic_contender <= best_heuristic:
            # If the best contender in this search loop produces a
            # better density ratio than the previous best, update best
            best_kde = kde_contender
            best_heuristic = heuristic_contender
            best_bandwidth = bandwidth_contender
            coarse_search = coarse_search/fine_search
        else:
            # Otherwise, current search window provided no better
            # bandwidth values, break from search loop
            break

    return best_kde


def _determine_high_low_freq_bins(labels, bin_count, padding_factor) -> tuple:
    """
    Calculate the contents of the highest and lowest frequency bins, as
    well as their index.
    Args:
        labels: The data labels to bin.
        bin_count: The number of bins to create.

    Returns:

    """
    label_min, label_max, step = get_label_bin_bounds(labels, bin_count, padding_factor)

    bins = [labels[(labels < label_min + step * (i+1)) & (labels >= label_min + step * i)] for i in range(bin_count)]
    high_freq = (bins[0], 0)
    for index, _bin in enumerate(bins):
        if _bin.shape[0] > high_freq[0].shape[0]:
            high_freq = (_bin, index)
    low_freq = (high_freq[0], high_freq[1])
    for index, _bin in enumerate(bins):
        if _bin.shape[0] != 0 and _bin.shape[0] <= low_freq[0].shape[0] and abs(high_freq[1] - low_freq[1]) < abs(high_freq[1] - index):
            low_freq = (_bin, index)

    return high_freq[0], high_freq[1], low_freq[0], low_freq[1]
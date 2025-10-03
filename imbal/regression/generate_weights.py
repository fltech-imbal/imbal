import numpy as np
from IPython.terminal.shortcuts.filters import eval_node
from matplotlib.figure import Figure
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from math import sqrt, pi, log, floor, ceil

def generate_weights(
        labels,
        samples_per_bin=100,
        bin_width=None, #TODO remove
        bin_count=None,
        density_mapping=None,
        bandwidth=None,
        steps_per_bin = 10,
        fine_search = 5,
        tolerance = 1e-3,
        return_kde=False,
        visualize_kde=False,
        save_figure=None,
        use_axes=None,
        verbose=False,
        padding_factor=0.01,
        optimization=None,
    ):
    """

    Args:
        labels:
        bin_count:
        density_mapping:
        bandwidth:
        steps_per_bin:
        fine_search:
        tolerance:
        return_kde:
        visualize_kde:
        save_figure:
        use_axes:
        verbose:
        padding_factor: Optional, default :code:`0.01`. Used to add a small padding to
            the data range used for binning. This padding should be specified as a percentage
            of the range of the data labels. This padding allows for a more graceful
            handling of scenarios where the minimum or maximum of the labels is the most
            frequent value in the labels.
        optimization:

    Returns:

    """

    if bin_count is None:
        bin_count = _determine_bin_count(labels, samples_per_bin, bin_width, padding_factor)

    if density_mapping is None:
        # Use KDE estimation to generate weights
        return _generate_density_mapping(
            labels,
            bandwidth=bandwidth,
            bin_count=bin_count,
            steps_per_bin=steps_per_bin,
            fine_search=fine_search,
            tolerance=tolerance,
            visualize_kde=visualize_kde,
            return_kde=return_kde,
            use_axes=use_axes,
            save_figure=save_figure,
            verbose=verbose,
            padding_factor=padding_factor,
            optimization=optimization
        )
    else:
        if isinstance(density_mapping, list):
            # Use uniform mapping for weights
            reweights = np.array(density_mapping) / np.sum(density_mapping)
        else:
            # Use function mapping for weights
            vectorized_function = np.vectorize(density_mapping)
            reweights = vectorized_function(labels)
            reweights = reweights / np.sum(reweights)
        return reweights

def _determine_bin_count(labels, samples_per_bin, bin_width, padding_factor):
    labels = np.sort(labels.reshape(-1,))
    if bin_width is None:
        return ceil(labels.shape[0] / samples_per_bin)
    else:
        label_min, label_max, _ = _get_label_bin_bounds(labels, 1, padding_factor)
        return ceil((label_max - label_min) / bin_width)



def _generate_density_mapping(
        labels,
        bandwidth=None,
        bin_count=None,
        steps_per_bin = 10,
        fine_search = 10,
        tolerance = 1e-3,
        visualize_kde=False,
        return_kde=False,
        use_axes=None,
        save_figure=None,
        verbose=False,
        padding_factor=0.01,
        optimization=None
) -> list:

    if bandwidth is None or bandwidth == 'binned' or bandwidth == 'binned_average' or bandwidth == 'binned_fit':
        # Use iterative, "binned-based" approach to approximate KDE
        kde = _iterative_kde_approximation(
            labels,
            bin_count=bin_count,
            bandwidth=bandwidth,
            steps_per_bin=steps_per_bin,
            fine_search=fine_search,
            tolerance=tolerance,
            verbose=verbose,
            padding_factor=padding_factor,
            optimization=optimization
        )
    else:
        # Use literal or explicit bandwidth to approximate KDE
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(labels.reshape(-1, 1))

    if verbose:
        print('Performing label to density to weight conversion...')

    approx = None
    if optimization == 'linear_interpolation':
        sample_points, sample_densities = _linearly_interpolate_kde(labels, kde, bin_count, steps_per_bin, padding_factor)
        approx = (sample_points, sample_densities)

        sample_densities = 1 / sample_densities
        reweights = np.interp(labels, sample_points, sample_densities)
    elif optimization == 'local':
        points, densities = _local_kde_optimization(labels, kde, steps_per_bin)
        approx = (points, densities)
        reweights = 1 / densities

    else:
        reweights = 1 / np.exp(kde.score_samples(labels.reshape(-1,1)).reshape(-1,))
    reweights = reweights / np.sum(reweights)
    if verbose:
        print('Conversion done')

    if visualize_kde or save_figure is not None or use_axes is not None:
        _plot_kde_graph(
            labels,
            bin_count,
            kde,
            visualize_kde=visualize_kde,
            verbose=verbose,
            padding_factor=padding_factor,
            kde_approximation=approx,
            save_figure=save_figure,
            use_axes=use_axes
        )

    return_values = [reweights]

    if return_kde:
        return_values.append(kde)

    if len(return_values) == 1:
        return return_values[0]
    else:
        return return_values

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



def _linearly_interpolate_kde(labels, kde, bin_count, steps_per_bin, padding_factor):
    total_samples = bin_count * steps_per_bin
    label_min, label_max, step = _get_label_bin_bounds(labels, bin_count, padding_factor)
    step /= steps_per_bin
    sample_points = np.array([label_min + i*step for i in range(total_samples + 1)])
    sample_densities = np.exp(kde.score_samples(sample_points.reshape(-1,1)).reshape(-1,))
    return sample_points, sample_densities

def _plot_kde_graph(
        labels,
        bin_count,
        kde,
        visualize_kde,
        verbose=False,
        padding_factor=0.01,
        kde_approximation=None,
        use_axes=None,
        save_figure=None
) -> None:
    if verbose:
        print('Plotting KDE...')

    labels = np.sort(labels.reshape(-1, ))
    high_freq_bin, high_freq_bin_index, low_freq_bin, low_freq_bin_index = _determine_high_low_freq_bins(labels,
                                                                                                         bin_count,
                                                                                                         padding_factor)
    label_min, label_max, step = _get_label_bin_bounds(labels, bin_count, padding_factor)
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
    # if kde_approximation is not None:
    #     plt.plot(kde_approximation[0], kde_approximation[1], label='Approximation', color='orange')

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
    if visualize_kde:
        plt.show()


def _iterative_kde_approximation(
    labels,
    bin_count=10,
    bandwidth=None,
    steps_per_bin=10,
    fine_search = 10,
    tolerance = 1e-3,
    verbose=False,
    padding_factor=0.01,
    optimization=None
) -> KernelDensity | tuple:
    if verbose:
        print('Starting bin-based KDE approximation...')

    # Data formatting
    labels = np.sort(labels.reshape(-1,))

    # Get bounds and bin width
    label_min, label_max, step = _get_label_bin_bounds(labels, bin_count, padding_factor)
    # Get bins with the highest frequency and lowest frequency
    high_freq_bin, high_freq_bin_index, low_freq_bin, low_freq_bin_index = _determine_high_low_freq_bins(labels, bin_count, padding_factor)
    desired_ratio = high_freq_bin.shape[0] / low_freq_bin.shape[0]

    # Generate lists used for even-spaced sampling across lowest and highest frequency bins
    spaced_high_freq_bin = np.linspace(high_freq_bin[0], high_freq_bin[0] + step, steps_per_bin).reshape(-1, 1)
    spaced_low_freq_bin = np.linspace(low_freq_bin[0], low_freq_bin[0] + step, steps_per_bin).reshape(-1, 1)

    spaced_even_curve = np.linspace(label_min, label_max, steps_per_bin*bin_count).reshape(-1, 1)
    print(spaced_even_curve.shape[0])
    bin_counts = [labels[(labels < label_min + step * (i + 1)) & (labels >= label_min + step * i)].shape[0] for i in range(bin_count)]
    even_counts = np.repeat(bin_counts, steps_per_bin)
    histogram_values = even_counts / step / labels.shape[0]
    print(histogram_values.shape[0])


    # Starting test ranges for bandwidth should be between 0.01*stddev and 3*stddev
    starting_bandwidth = float(np.std(labels)) * 1.49
    coarse_search = starting_bandwidth

    # Track best results during iteration
    best_kde = None
    best_heuristic = None
    best_bandwidth = starting_bandwidth

    average_mode = bandwidth == 'binned_average'
    fit_mode = bandwidth == 'binned_fit'
    labels = labels.reshape(-1, 1)

    # Ensure at least one loop, and loop until ratio is within tolerance,
    # or search becomes too fine grain (perhaps ideal ratio is impossible)
    while best_kde is None or (abs(desired_ratio - best_heuristic) > tolerance and coarse_search / fine_search > 1e-6):
        search_steps = round(fine_search)
        kde_contender = None
        heuristic_contender = None
        bandwidth_contender = None

        for i in range(search_steps * 2 + 1):
            current_bandwidth = best_bandwidth + (i - search_steps) * coarse_search / fine_search
            # Prevent negative and zero bandwidths
            if current_bandwidth <= 1e-6:
                continue

            # Fit KDE with current bandwidth
            kde = KernelDensity(bandwidth=current_bandwidth)
            kde.fit(labels)

            if average_mode:
                high_densities = np.exp(kde.score_samples(spaced_high_freq_bin))
                low_densities = np.exp(kde.score_samples(spaced_low_freq_bin))
                heuristic = abs(float(np.mean(high_densities)) / float(np.mean(low_densities)) - desired_ratio)
            elif fit_mode:
                # CURVE FIT BASED HEURISTIC
                kde_densities = np.exp(kde.score_samples(spaced_even_curve))
                heuristic = np.mean(np.abs(histogram_values - kde_densities))
                print(current_bandwidth, heuristic)
            else:
                # DENSITY-RATIO BASED HEURISTIC
                high_densities = np.exp(kde.score_samples(spaced_high_freq_bin))
                low_densities = np.exp(kde.score_samples(spaced_low_freq_bin))
                max_area = (-high_densities[0]/2 - high_densities[-1]/2 + np.sum(high_densities)) * step/steps_per_bin
                min_area = (-low_densities[0]/2 - low_densities[-1]/2 + np.sum(low_densities)) * step/steps_per_bin
                heuristic = abs(max_area / min_area - desired_ratio)
                print(current_bandwidth, max_area, min_area, heuristic)

            if bandwidth_contender is None or heuristic < heuristic_contender:
                print('contender', current_bandwidth, heuristic)
                kde_contender = kde
                heuristic_contender = heuristic
                bandwidth_contender = current_bandwidth


        if best_heuristic is None or heuristic_contender <= best_heuristic:
            # If the best contender in this search loop produces a
            # better density ratio than the previous best, update best
            print('best', bandwidth_contender, heuristic_contender)
            if verbose:
                print(f'Current bandwidth approximation: {bandwidth_contender:.4}, absolute ratio difference: {heuristic_contender:.4}')
            best_kde = kde_contender
            best_heuristic = heuristic_contender
            best_bandwidth = bandwidth_contender
            coarse_search = coarse_search/fine_search
        else:
            # Otherwise, current search window provided no better
            # bandwidth values, break from search loop
            break

    if verbose:
          print(f'Final bandwidth approximation: {best_bandwidth:.4}, absolute ratio difference: {abs(best_heuristic - desired_ratio):.4}')
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
    label_min, label_max, step = _get_label_bin_bounds(labels, bin_count, padding_factor)

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

def _get_label_bin_bounds(labels, bin_count, padding_factor) -> tuple:
    label_min = labels[0]
    label_max = labels[-1] + 1e-6
    label_range = label_max - label_min
    label_min = label_min - label_range*padding_factor
    label_max = label_max + label_range*padding_factor
    step = float((label_max - label_min) / bin_count)
    return label_min, label_max, step

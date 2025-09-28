import numpy as np
from matplotlib.figure import Figure
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

def generate_weights(
        labels,
        bin_count=32,
        density_mapping=None,
        bandwidth=None,
        samples_per_bin = 10,
        fine_search = 10,
        tolerance = 1e-3,
        return_kde=False,
        visualize_kde=False,
        return_figure=False,
        verbose=False
    ):

    if density_mapping is None:
        # Use KDE estimation to generate weights
        return _generate_density_mapping(
            labels,
            bandwidth=bandwidth,
            bin_count=bin_count,
            samples_per_bin=samples_per_bin,
            fine_search=fine_search,
            tolerance=tolerance,
            visualize_kde=visualize_kde,
            return_kde=return_kde,
            return_figure=return_figure,
            verbose=verbose
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

def _generate_density_mapping(
        labels,
        bandwidth=None,
        bin_count=32,
        samples_per_bin = 50,
        fine_search = 10,
        tolerance = 1e-3,
        visualize_kde=False,
        return_kde=False,
        return_figure=False,
        verbose=False
) -> list:
    """
    TODO

    Args:
        labels:
        bandwidth:
        bin_count:
        samples_per_bin:
        fine_search:
        tolerance:
        visualize_kde:
        return_kde:
        return_figure:

    Returns:

    """
    if bandwidth is None or bandwidth == 'binned' or bandwidth == 'binned_average':
        # Use iterative, "binned-based" approach to approximate KDE
        kde = _iterative_kde_approximation(
            labels,
            bin_count=bin_count,
            bandwidth=bandwidth,
            samples_per_bin=samples_per_bin,
            fine_search=fine_search,
            tolerance=tolerance,
            verbose=verbose
        )
    else:
        # Use literal or explicit bandwidth to approximate KDE
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(labels.reshape(-1, 1))

    if verbose:
        print('Performing label to density to weight conversion...')
    reweights = 1 / np.exp(kde.score_samples(labels.reshape(-1,1)).reshape(-1,))
    reweights = reweights / np.sum(reweights)
    if verbose:
        print('Conversion done')

    fig = None
    if visualize_kde or return_figure:
        fig = _plot_kde_graph(labels, bin_count, kde, visualize_kde=visualize_kde, verbose=verbose)

    return_values = [reweights]

    if return_kde:
        return_values.append(kde)
    if return_figure:
        return_values.append(fig)

    if len(return_values) == 1:
        return return_values[0]
    else:
        return return_values

def _plot_kde_graph(labels, bin_count, kde, visualize_kde, verbose=False) -> Figure:
    high_freq_bin, high_freq_bin_index, low_freq_bin, low_freq_bin_index = _determine_high_low_freq_bins(labels,
                                                                                                         bin_count)

    print('Plotting KDE...')

    labels = np.sort(labels.reshape(-1, ))
    label_min = float(labels[0])
    label_max = float(labels[-1] + 1e-6)
    step = float((label_max - label_min) / bin_count)
    min_count = low_freq_bin.shape[0]
    max_count = high_freq_bin.shape[0]

    x_plot = np.linspace(label_min - 1, label_max + 1, 1000).reshape(-1, 1)
    log_dens = kde.score_samples(x_plot)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x_plot, np.exp(log_dens), label='KDE Curve')
    plt.hist(labels, bins=[label_min + i * step for i in range(bin_count)], density=True, alpha=0.6, label='Histogram')

    f_min_bar = label_min + (low_freq_bin_index + .5) * step
    plt.axvline(x=f_min_bar, color='red', linestyle='--', linewidth=2)
    plt.text(f_min_bar, plt.ylim()[1] * 0.02, f'f_min = {min_count}',
             rotation=90, color='red',
             verticalalignment='bottom', horizontalalignment='right')

    f_max_bar = label_min + (high_freq_bin_index + .5) * step
    plt.axvline(x=f_max_bar, color='red', linestyle='--', linewidth=2)
    plt.text(f_max_bar, plt.ylim()[1] * 0.02, f'f_max = {max_count}',
             rotation=90, color='red',
             verticalalignment='bottom', horizontalalignment='right')

    plt.title(
        f'Kernel Density Estimation (f_max/f_min = {max_count / min_count:.1f}, bandwidth = {kde.bandwidth_:.3f})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    if visualize_kde:
        plt.show()

    return fig

def _iterative_kde_approximation(
    labels,
    bin_count=10,
    bandwidth=None,
    samples_per_bin = 50,
    fine_search = 10,
    tolerance = 1e-3,
    verbose=False
) -> KernelDensity | tuple:
    if verbose:
        print('Starting bin-based KDE approximation...')

    # Data formatting
    labels = np.sort(labels.reshape(-1,))

    # Get bounds and bin width
    label_min, label_max, step = _get_label_bin_bounds(labels, bin_count)
    # Get bins with the highest frequency and lowest frequency
    high_freq_bin, high_freq_bin_index, low_freq_bin, low_freq_bin_index = _determine_high_low_freq_bins(labels, bin_count)
    desired_ratio = high_freq_bin.shape[0] / low_freq_bin.shape[0]

    # Generate lists used for even-spaced sampling across lowest and highest frequency bins
    spaced_high_freq_bin = np.array([high_freq_bin[0] + i / samples_per_bin*step for i in range(samples_per_bin)]).reshape(-1, 1)
    spaced_low_freq_bin = np.array([low_freq_bin[0] + i / samples_per_bin*step for i in range(samples_per_bin)]).reshape(-1, 1)

    # Starting test ranges for bandwidth should be between 0.01*stddev and 3*stddev
    starting_bandwidth = float(np.std(labels)) * 1.49
    coarse_search = starting_bandwidth

    best_kde = None
    best_density_ratio = 0
    best_bandwidth = starting_bandwidth

    average_mode = bandwidth == 'binned_average'
    labels = labels.reshape(-1, 1)

    while best_kde is None or abs(desired_ratio - best_density_ratio) > tolerance:
        search_steps = round(fine_search)
        kde_contender = None
        density_ratio_contender = None
        bandwidth_contender = None

        for i in range(search_steps * 2 + 1):
            current_bandwidth = best_bandwidth + (i - search_steps) * coarse_search / fine_search

            # Prevent negative and zero bandwidths
            if current_bandwidth <= 1e-6:
                continue

            # Fit KDE with current bandwidth
            kde = KernelDensity(bandwidth=current_bandwidth)
            kde.fit(labels)

            # Compute
            high_densities = np.exp(kde.score_samples(spaced_high_freq_bin))
            low_densities = np.exp(kde.score_samples(spaced_low_freq_bin))

            if average_mode:
                density_ratio = float(np.mean(high_densities)) / float(np.mean(low_densities))
            else:
                max_area = (-high_densities[0]/2 - high_densities[-1]/2 + np.sum(high_densities)) * step/samples_per_bin
                min_area = (-low_densities[0]/2 - low_densities[-1]/2 + np.sum(low_densities)) * step/samples_per_bin
                density_ratio = max_area / min_area
            if bandwidth_contender is None or abs(density_ratio - desired_ratio) < abs(density_ratio_contender - desired_ratio):
                kde_contender = kde
                density_ratio_contender = density_ratio
                bandwidth_contender = current_bandwidth


        if abs(density_ratio_contender - desired_ratio) < abs(best_density_ratio - desired_ratio):
            # If the best contender in this search loop produces a
            # better density ratio than the previous best, update best
            if verbose:
                print(f'Current bandwidth approximation: {bandwidth_contender:.4}, absolute ratio difference: {abs(density_ratio_contender - desired_ratio):.4}')
            best_kde = kde_contender
            best_density_ratio = density_ratio_contender
            best_bandwidth = bandwidth_contender
            coarse_search = coarse_search/fine_search
        else:
            # Otherwise, current search window provided no better
            # bandwidth values, break from search loop
            break

    print(f'Final bandwidth approximation: {best_bandwidth:.4}, absolute ratio difference: {abs(best_density_ratio - desired_ratio):.4}')
    return best_kde


def _determine_high_low_freq_bins(labels, bin_count) -> tuple:
    """
    Calculate the contents of the highest and lowest frequency bins, as
    well as their index.
    Args:
        labels: The data labels to bin.
        bin_count: The number of bins to create.

    Returns:

    """
    label_min, label_max, step = _get_label_bin_bounds(labels, bin_count)

    bins = [labels[(labels < label_min + step * (i+1)) & (labels >= label_min + step * i)] for i in range(bin_count)]
    low_freq = (bins[0], 0)
    high_freq = (bins[0], 0)
    for index, _bin in enumerate(bins):
        if _bin.shape[0] != 0 and _bin.shape[0] < low_freq[0].shape[0]:
            low_freq = (_bin, index)
        if _bin.shape[0] > high_freq[0].shape[0]:
            high_freq = (_bin, index)

    return high_freq[0], high_freq[1], low_freq[0], low_freq[1]

def _get_label_bin_bounds(labels, bin_count) -> tuple:
    label_min = labels[0]
    label_max = labels[-1] + 1e-6
    step = float((label_max - label_min) / bin_count)
    return label_min, label_max, step

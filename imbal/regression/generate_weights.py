import numpy as np
from sklearn.neighbors import KernelDensity
from enum import Enum
import matplotlib.pyplot as plt

class RegressionWeightMode(Enum):
    AVERAGE = 0
    AUC = 1

def generate_weights(
        labels,
        bin_count=10,
        weight_mapping=None,
        mode=RegressionWeightMode.AUC,
        bin_sample_count = 50,
        fine_search = 10,
        tolerance = 1e-3,
        return_kde=False,
        visualize_kde=False
    ):

    # Data formatting
    labels = np.sort(labels.reshape(-1,))
    label_min = labels[0]
    label_max = labels[-1] + 1e-6
    step = float((label_max - label_min) / bin_count)

    # Calculate highest and lowest frequency bin
    bins = [labels[(labels < label_min + step * (i+1)) & (labels >= label_min + step * i)] for i in range(bin_count)]
    high_freq_bin = bins[0]
    low_freq_bin = bins[0]
    for _bin in bins:
        if _bin.shape[0] != 0 and _bin.shape[0] < low_freq_bin.shape[0]:
            low_freq_bin = _bin
        if _bin.shape[0] > high_freq_bin.shape[0]:
            high_freq_bin = _bin

    desired_ratio = high_freq_bin.shape[0] / low_freq_bin.shape[0]

    # Generate
    spaced_high_freq_bin = np.array([high_freq_bin[0] + i / bin_sample_count*step for i in range(bin_sample_count)]).reshape(-1, 1)
    spaced_low_freq_bin = np.array([low_freq_bin[0] + i / bin_sample_count*step for i in range(bin_sample_count)]).reshape(-1, 1)

    starting_bandwidth = np.std(labels) * 1.49
    coarse_search = starting_bandwidth

    best_kde = None
    best_density_ratio = 0
    best_bandwidth = starting_bandwidth

    while best_kde is None or abs(desired_ratio - best_density_ratio) > tolerance:
        search_steps = round(fine_search)
        kde_contender = None
        density_ratio_contender = None
        bandwidth_contender = None
        for i in range(search_steps * 2 + 1):
            if (best_bandwidth + (i - search_steps) * coarse_search/fine_search) <= 1e-6:
                continue
            current_bandwidth = best_bandwidth + (i - search_steps) * coarse_search/fine_search
            kde = KernelDensity(bandwidth=current_bandwidth)
            kde.fit(labels.reshape(-1, 1))

            max_densities = np.exp(kde.score_samples(spaced_high_freq_bin))
            min_densities = np.exp(kde.score_samples(spaced_low_freq_bin))
            if mode == RegressionWeightMode.AVERAGE:
                if np.mean(min_densities) != 0:
                    density_ratio = float(np.mean(max_densities)) / float(np.mean(min_densities))
                else:
                    continue
            else:
                max_area = (max_densities[0]/2 + max_densities[-1]/2 + np.sum(max_densities[1:-1])) * step/bin_sample_count
                min_area = (min_densities[0]/2 + min_densities[-1]/2 + np.sum(min_densities[1:-1])) * step/bin_sample_count
                density_ratio = max_area / min_area
            if bandwidth_contender is None or abs(density_ratio - desired_ratio) < abs(density_ratio_contender - desired_ratio):
                kde_contender = kde
                density_ratio_contender = density_ratio
                bandwidth_contender = current_bandwidth

        if abs(density_ratio_contender - desired_ratio) < abs(best_density_ratio - desired_ratio):
            best_kde = kde_contender
            best_density_ratio = density_ratio_contender
            best_bandwidth = bandwidth_contender

            coarse_search = coarse_search/fine_search
        else:
            break

    reweights = 1 / np.exp(best_kde.score_samples(labels.reshape(-1,1)).reshape(-1,))
    reweights = reweights / np.sum(reweights)

    if weight_mapping is not None:
        reweights = np.vectorize(weight_mapping)(reweights)
        reweights = reweights / np.sum(reweights)

    if visualize_kde:
        x_plot = np.linspace(label_min - 1, label_max + 1, 1000).reshape(-1, 1)
        log_dens = best_kde.score_samples(x_plot)
        plt.figure(figsize=(8, 6))
        plt.plot(x_plot, np.exp(log_dens), label='KDE Curve')
        plt.hist(labels, bins=bin_count, density=True, alpha=0.6, label='Histogram')
        plt.title('Kernel Density Estimation')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

    if return_kde:
        return reweights, best_kde
    else:
        return reweights


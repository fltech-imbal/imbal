import numpy as np
from sklearn.neighbors import KernelDensity
from enum import Enum

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
        return_kde=False
    ):

    labels = labels.reshape(-1, 1)
    row_labels = labels.reshape(-1, )
    row_labels = np.sort(row_labels)
    label_min = row_labels[0]
    label_max = row_labels[-1] + 1e-6
    step = float((label_max - label_min) / bin_count)

    low_bin = labels[labels <= label_min + step].reshape(-1, 1)
    high_bin = labels[labels >= label_max - step].reshape(-1, 1)
    desired_ratio = low_bin.shape[0] / high_bin.shape[0]
    # print('Desired', desired_ratio)

    low_even_bin = np.array([label_min + i / bin_sample_count*step for i in range(bin_sample_count)]).reshape(-1, 1)
    high_even_bin = np.array([label_max - i / bin_sample_count*step for i in range(bin_sample_count)]).reshape(-1, 1)

    starting_bandwidth = np.std(row_labels) * 1.49
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
            if (best_bandwidth + (i - search_steps) * coarse_search/fine_search) <= 0:
                continue
            current_bandwidth = best_bandwidth + (i - search_steps) * coarse_search/fine_search
#             print(current_bandwidth)
            kde = KernelDensity(bandwidth=current_bandwidth)
            kde.fit(labels)

            max_densities = np.exp(kde.score_samples(low_even_bin))
            min_densities = np.exp(kde.score_samples(high_even_bin))
            if mode == RegressionWeightMode.AVERAGE:
                if np.mean(min_densities) != 0:
                    density_ratio = float(np.mean(max_densities)) / float(np.mean(min_densities))
                else:
                    continue
            else:
                max_area = np.sum((max_densities[1:] + max_densities[:-1])/2) * step/bin_sample_count
                min_area = np.sum((min_densities[1:] + min_densities[:-1])/2) * step / bin_sample_count
                density_ratio = max_area / min_area
#                 print('AUC', density_ratio)
            if bandwidth_contender is None or abs(density_ratio - desired_ratio) < abs(density_ratio_contender - desired_ratio):
                kde_contender = kde
                density_ratio_contender = density_ratio
                bandwidth_contender = current_bandwidth
#                 print('Best', bandwidth_contender)
#             print()

        if abs(density_ratio_contender - desired_ratio) < abs(best_density_ratio - desired_ratio):
            best_kde = kde_contender
            best_density_ratio = density_ratio_contender
            best_bandwidth = bandwidth_contender
            coarse_search /= search_steps
            fine_search /= search_steps
        else:
            break

    reweights = 1 / np.exp(best_kde.score_samples(labels).reshape(-1,))
    reweights = reweights / np.sum(reweights)

    if weight_mapping is not None:
        reweights = np.vectorize(weight_mapping)(reweights)
        reweights = reweights / np.sum(reweights)

    if return_kde:
        return reweights, best_kde
    else:
        return reweights


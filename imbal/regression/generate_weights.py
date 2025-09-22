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
        mode = RegressionWeightMode.AVERAGE,
        starting_bandwidth=1,
        coarse_search = 1,
        fine_search = 0.1,
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

    best_kde = None
    best_density_ratio = 0
    best_bandwidth = starting_bandwidth

    while best_kde is None or abs(desired_ratio - best_density_ratio) > tolerance:
        search_steps = round(coarse_search / fine_search)
        kde_contender = None
        density_ratio_contender = None
        bandwidth_contender = None
        for i in range(search_steps * 2 + 1):
            if (best_bandwidth + (i - search_steps) * fine_search) <= 0:
                continue
            current_bandwidth = best_bandwidth + (i - search_steps) * fine_search
#             print(current_bandwidth)
            kde = KernelDensity(bandwidth=current_bandwidth)
            kde.fit(labels)
            if mode == RegressionWeightMode.AVERAGE:
                max_density = float(np.mean(np.exp(kde.score_samples(low_bin))))
                min_density = float(np.mean(np.exp(kde.score_samples(high_bin))))
                density_ratio = max_density / min_density
#                 print(density_ratio)
            else:
                # TODO: AUC METHOD
                max_density = float(np.mean(np.exp(kde.score_samples(low_bin))))
                min_density = float(np.mean(np.exp(kde.score_samples(high_bin))))
                density_ratio = max_density / min_density
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


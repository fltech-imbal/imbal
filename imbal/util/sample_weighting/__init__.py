import numpy as np
from math import ceil

def get_label_bin_bounds(labels, bin_count, padding_factor) -> tuple:
    labels = np.asarray(labels)

    # Handle when list is passed as a 1D array, instead of as a column vector
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    label_min = labels.min(axis=0)
    label_max = labels.max(axis=0) + 1e-6
    label_range = label_max - label_min
    label_min = label_min - label_range * padding_factor
    label_max = label_max + label_range * padding_factor
    step = (label_max - label_min) / bin_count
    return label_min, label_max, step

def calculate_bin_count(labels, bin_count, average_samples_per_bin) -> int:
    if bin_count is None:
        labels = np.sort(labels.reshape(-1, ))
        return ceil(labels.shape[0] / average_samples_per_bin)
    else:
        return bin_count
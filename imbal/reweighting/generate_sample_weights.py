import numpy as np
from numpy.typing import NDArray

def generate_sample_weights(class_labels, weight_mapping, mode='classification') -> NDArray:
    unique_classes, unique_counts = np.unique(class_labels.reshape(-1,), return_counts=True)
    full_weight_mapping = {}
    balanced_mapping = {}
    weight_sum = 0

    if isinstance(weight_mapping, dict):
        for cls in unique_classes:
            if cls in weight_mapping:
                full_weight_mapping[cls] = weight_mapping[cls]
            else:
                full_weight_mapping[cls] = 1
            weight_sum += full_weight_mapping[cls]
    else:
        if len(weight_mapping) != len(unique_classes):
            raise ValueError('When passing weights as a list, the length of the list of weights must be equal to the number of classes.')
        for cls, weight in zip(unique_classes, weight_mapping):
            full_weight_mapping[cls] = weight
            weight_sum += weight


    for label, count  in zip(unique_classes, unique_counts):
        balanced_mapping.update({label: full_weight_mapping[label] / weight_sum / count})

    return np.array([balanced_mapping[label] for label in class_labels])
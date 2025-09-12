import numpy as np

def class_to_sample_weights(class_labels, weight_mapping):
    unique_classes, unique_counts = np.unique(class_labels, return_counts=True)
    balanced_mapping = {}
    weight_sum = 0
    for cls in unique_classes:
        if cls not in weight_mapping:
            weight_mapping[cls] = 1
        weight_sum += weight_mapping[cls]

    for label, count  in zip(unique_classes, unique_counts):
        balanced_mapping.update({label: weight_mapping[label] / weight_sum / count})

    return [balanced_mapping[label] for label in class_labels]
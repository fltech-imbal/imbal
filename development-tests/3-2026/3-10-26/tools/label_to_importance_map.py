

def map_labels_to_importance_weights(labels, importance_weights) -> dict:
    """
    Map labels to their corresponding importance weights.

    Parameters:
    - labels (np.ndarray): An array of labels.
    - importance_weights (np.ndarray): An array of importance weights corresponding to each label.

    Returns:
    - dict: A dictionary where each key is a label and its value is the corresponding importance weight.
    """
    if len(labels) != len(importance_weights):
        raise ValueError("Labels and importance weights must be of the same length.")

    label_to_importance_weight_mapping = {}
    for label, importance_weight in zip(labels, importance_weights):
        label_to_importance_weight_mapping[float(label)] = float(importance_weight)

    return label_to_importance_weight_mapping
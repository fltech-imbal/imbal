import numpy as np

def generate_weights(
        labels,
        weight_mapping=None
    ):
    """
    Generates a list of weights, where the index of each weight corresponds to the label
    at the index of the provides list of labels. The sum of all weights in the returned
    list of weights will be normalized to 1.

    Args:
        labels: A NumPy array of labels, arranged as a column vector
        weight_mapping: A dictionary or list of mappings from class label to weight. If
            a dictionary is provided, keys will be interpreted as class labels, and the corresponding
            values interpreted as the fraction of the final weight the class should take up. If a
            list is provided, the entries in the list will be assumed the fraction of the final weight
            the class should take up, sorted in ascending order by classes present in :code:`labels`.

    Returns:
        A normalized list of weights, where the index of each weight corresponds to the label
        at the index of the provides list of labels.

    Example:

    .. code-block:: python

        >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]).reshape(-1,1)
        >>> labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1]).reshape(-1,1)

        >>> weights = generate_weights(labels, { 0: 0.6, 1: 0.4 })

        >>> print(weights)
        [0.075 0.075 0.075 0.075 0.075 0.075 0.075 0.075 0.2 0.2]

    .. code-block:: python

        >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1,1)
        >>> labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2]).reshape(-1,1)

        >>> weights = generate_weights(labels, { 0: 0.4, 1: 0.3, 2: 0.3 })

        >>> print(weights)
        [0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.1 0.1 0.1 0.15 0.15]
    """
    labels = labels.reshape(-1, )
    unique_classes, unique_counts = np.unique(labels, return_counts=True)
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
        if weight_mapping is None:
            for cls in unique_classes:
                full_weight_mapping[cls] = 1
                weight_sum += full_weight_mapping[cls]
        else:
            if len(weight_mapping) != len(unique_classes):
                raise ValueError(
                    'When passing weights as a list, the length of the list of weights must be equal to the number of classes.')
            for cls, weight in zip(unique_classes, weight_mapping):
                full_weight_mapping[cls] = weight
                weight_sum += weight

    for label, count in zip(unique_classes, unique_counts):
        balanced_mapping.update({label: full_weight_mapping[label] / weight_sum / count})

    return np.array([balanced_mapping[label] for label in labels])


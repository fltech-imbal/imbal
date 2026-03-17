import numpy as np
from imbal.util.backend.tools import verify_weight_scale

def generate_sample_weights(
        labels,
        class_weights=None
    ):
    """
    Generates a list of weights, where the index of each weight corresponds to the label
    at the index of the provides list of labels. The sum of all weights in the returned
    list of weights will be normalized to :math:`n`.

    Normally, it is standard to normalize weights to :math:`1`. However, when no weights
    are provided to Tensorflow, its default behavior is to assign a weight of :math:`1`
    to each sample, meaning the total weight for the dataset is :math:`n`. Straying from
    this pattern would affect the scale of calculated loss values, which would also
    have an impact on how learning rates perform, therefore we have decided to align
    our weight generation implementations as closely as possible with Tensorflow's
    default behavaiors.

    Args:
        labels: A NumPy array of labels, arranged as a row vector, column vector, or list of one-hot vectors.
        class_weights: A dictionary or list of mappings from class label to weight. If
            no weight mapping is provided, each class will be weighted equally (samples of
            more frequent classes will be weighted lower, and vice versa). If
            a dictionary is provided, keys will be interpreted as class labels, and the corresponding
            values interpreted as the fraction of the final weight the class should take up. If a
            list is provided, the entries in the list will be assumed the fraction of the final weight
            the class should take up, sorted in ascending order by classes present in :code:`labels`.

    Returns:
        A normalized list of weights, where the index of each weight corresponds to the label
        at the index of the provides list of labels.

    Example:

    .. code-block:: python

        >>> import imbal
        >>> import numpy as np

        >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]).reshape(-1,1)
        >>> labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1]).reshape(-1,1)

        >>> weights = imbal.classification.generate_sample_weights(labels, { 0: 0.6, 1: 0.4 })

        >>> print(weights)
        [0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 2.0 2.0]

    .. code-block:: python

        >>> import imbal
        >>> import numpy as np

        >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1,1)
        >>> labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2]).reshape(-1,1)

        >>> weights = imbal.classification.generate_sample_weights(labels, { 0: 0.4, 1: 0.3, 2: 0.3 })

        >>> print(weights)
        [0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 1.3 1.3 1.3 1.95 1.95]
    """

    if labels.ndim == 2 and labels.shape[1] != 1:
        labels = labels.argmax(axis=1)

    labels = labels.reshape(-1, )
    unique_classes, unique_counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique_classes, unique_counts))
    full_weight_mapping = {}
    balanced_mapping = {}
    weight_sum = 0

    multi_weight = False
    weights = None

    if isinstance(class_weights, dict):
        weights = np.vectorize(
            lambda x: class_weights.get(x, 1) / class_counts.get(x)
        )(labels)
    else:
        if class_weights is None:
            for cls in unique_classes:
                full_weight_mapping[cls] = 1
                weight_sum += full_weight_mapping[cls]
        else:
            class_weights = np.array(class_weights)
            if class_weights.ndim == 1:
                if len(class_weights) != len(unique_classes):
                    raise ValueError(
                        'When passing weights as a list, the length of the list of weights must be equal to the number of classes.')
            else:
                assert class_weights.ndim == 2
                for cls, weight in zip(unique_classes, class_weights):
                    full_weight_mapping[cls] = weight
                    weight_sum += weight


        for label, count in zip(unique_classes, unique_counts):
            balanced_mapping.update({label: full_weight_mapping[label] / weight_sum / count * labels.shape[0]})
        weights = np.array([balanced_mapping[label] for label in labels])


    weights = verify_weight_scale(weights, show_warning=False, axis=1 if multi_weight else None)
    return weights


import numpy as np
from imbal.util.backend.tools import verify_weight_scale

def generate_sample_weights(
        labels,
        class_weight=None
    ):
    """
    Generates a list of sample weights for the provided labels, such that the sum
    of all weights in the returned list of weights will be normalized to :math:`n`, and
    such that the sum of each sample weight of a given class is proportional to some
    class weight ratio (ex. 50/50 for binary labels).

    .. note::

       A list of samples weights of size :math:`n` is normalized to sum to :math:`n`, since
       by default TensorFlow's samples weights are :math:`1` per sample. This normalization
       ensures that average sample weight is still :math:`1`, while still addressing class
       imbalance.

    Args:
        labels: A NumPy array of labels, arranged as a row vector, a :math:`n` by :math:`1` column vector,
            or list of one-hot vectors.
        class_weight: Optional, default :code:`None`. A dictionary or list of mappings from class label to
            weight. If unspecified, each class will be weighted equally (ex. 50/50 for binary labels). If
            a dictionary is provided, keys will be interpreted as class labels, and the corresponding
            values interpreted as the proportion of the final sum weight the class should take up. If a
            list is provided, the entries in the list will be assumed the proportion of the final sum weight
            the class should take up, sorted in ascending order by classes present in :code:`labels`.

    Returns:
        A list of weights, where the index of each weight corresponds to the label
        at the index of the provides list of labels.

    Example:

    .. code-block:: python

        >>> import imbal
        >>> import numpy as np

        >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]).reshape(-1,1)
        >>> labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1]).reshape(-1,1)

        >>> weights = imbal.classification.generate_sample_weights(labels, { 0: 0.6, 1: 0.4 })

        >>> print(weights)
        [0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 2.   2.  ]

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

    if isinstance(class_weight, dict):
        weights = np.vectorize(
            lambda x: class_weight.get(x, 1) / class_counts.get(x)
        )(labels)
    else:
        if class_weight is None:
            for cls in unique_classes:
                full_weight_mapping[cls] = 1
                weight_sum += full_weight_mapping[cls]
        else:
            class_weights = np.array(class_weight)
            if class_weights.ndim == 1:
                if len(class_weights) != len(unique_classes):
                    raise ValueError(
                        'When passing weights as a list, the length of the list of weights must be equal to the number of classes.')
                for i in range(len(class_weights)):
                    full_weight_mapping[i] = class_weights[i]
                    weight_sum += class_weights[i]
            else:
                class_weights = np.array(class_weights)

                if class_weights.shape[1] != len(unique_classes):
                    raise ValueError(
                        "2D class_weights must have shape (k, num_classes)"
                    )

                k = class_weights.shape[0]
                n = labels.shape[0]

                weights = np.zeros((k, n))

                for i in range(k):
                    weight_sum = np.sum(class_weights[i])

                    balanced_mapping = {}
                    for cls, count in zip(unique_classes, unique_counts):
                        cls_weight = class_weights[i][np.where(unique_classes == cls)[0][0]]
                        balanced_mapping[cls] = cls_weight / weight_sum / count * n

                    weights[i] = np.array([balanced_mapping[label] for label in labels])
                return weights


        for label, count in zip(unique_classes, unique_counts):
            balanced_mapping.update({label: full_weight_mapping[label] / weight_sum / count * labels.shape[0]})
        weights = np.array([balanced_mapping[label] for label in labels])

    weights = verify_weight_scale(weights, show_warning=False)
    return weights


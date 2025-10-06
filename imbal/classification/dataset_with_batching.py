from imbal.util.stratified_sampling.dataset_with_batching import DatasetWithBatching as UtilDataset
from imbal.util.constants import ModelType

class DatasetWithBatching(UtilDataset):
    """
    An extension of `TensorFlow's PyDataset class <https://www.tensorflow.org/api_docs/python/tf/keras/utils/PyDataset>`_.
    This class can be used to ensure that classification data remains stratified during
    the batching process commonly used for training.

    To stratify, data-label pairs are checked to
    ensure that there is a sufficient number of samples of each class such that
    every batch contains at least one instance of each class. In the event
    that a particular class does not contain enough samples for this to be true,
    the samples in that class will be duplicated until a sufficient number of
    samples is reached. Then, the sample weights of each sample in the class is
    adjusted to account for the duplication, such that the sum of the weights of
    all copies of a particular sample is equal to the weight of the original
    singular sample (ex. 3 copies of a samples with weight 0.3 will have adjusted
    weights of 0.1 each).

    Once all classes have the property above, samples are distributed across each batch via
    rotation, such that every batch will
    have as close to the same number of samples as possible (a maximum
    difference of one sample) and the sum of the weights of the samples in
    each batch is approximately equal (the difference in weights is minimized
    the more samples there are in each batch).

    **Note:** Where appropriate, documentation for functions from :code:`tf.keras.PyDataset` has been
    overridden to be more descriptive. Any other non-descriptive documentation of individual functions
    on this page is due to a lack of documentation in TensorFlow's original source code. Still, TensorFlow's
    documentation and source code for the :code:`PyDataset` class can be found `here <https://www.tensorflow.org/api_docs/python/tf/keras/utils/PyDataset>`_.

    Args:
        x_set: A NumPy array of data points, arranged as a column vector
        y_set: A NumPy array of labels, arranged as a column vector
        sample_weights: Optional, default :code:`None`. A NumPy array of weights,
            arranged as a column vector. When :code:`None`, all samples are assumed to be equally weighted.
        batch_size: Optional, default :code:`64`. The approximate size of each batch.
            This value is used as a guideline, actual batch size may vary since the stratification
            process affects the number of data points to be batched.
        num_batches: Optional, default :code:`None`. The number of batches to be generated after
            stratification. If specified, overrides the value of :code:`batch_size`.
        seed: Optional, default :code:`0`. The random seed for batch randomization.
        shuffle: Optional, default :code:`True`. Whether data should be shuffled during batching and between epochs.

    Each batch in the :code:`StratifiedBatcher` is stored as a tuple of the form
    :code:`(batch_x, batch_y, batch_weights)`. In this format, batches can be
    retrieved then manually fed to TensorFlow's :code:`model.fit()` or :code:`model.predict()`,
    but TensorFlow also allows for children of the :code:`PyDataset` class to be
    passed to its models as well.

    After instansiation, the number of batches generated can be retrieved by
    calling the :code:`len()` function on the :code:`StratifiedBatcher` object,
    as shown in the example below. Additionally, batches can be retrieved manually
    simply by indexing the object, such as :code:`sampler[i]`.

    In the example that follows, we see a dataset of 10 unique data points, from 0 to 9.
    Majority of the instances are of class 0, as indicated by the labels, but data point 9
    is of class 1, making it an instance of a much rarer class.

    The example calls for the creation of two batches, however, there is only one instance
    of a sample in class 1. Therefore, the members of class 1 will be duplicated, and the
    weights of its members halved to account for the fact that there is now 2 instances of
    class 1's members.

    The results in the example show how the duplicated class 1 members results in a batch
    of length 6 and a batch of length 5, each containing an instance of data point 9.
    Additionally, each data point is weighted evenly, but copy of data point 9 has half
    of the weight compared to the others, to compensate for the fact it has been duplicated.

    Example:

    .. code-block:: python

        >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]).reshape(-1,1)
        >>> labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(-1,1)

        >>> sampler = DatasetWithBatching(data, labels, num_batches=2)

        >>> # Batched data
        >>> print(sampler[0][0], sampler[1][0])
        [[6] [9] [7] [0] [5] [8]] [[9] [1] [2] [3] [4]]

        >>> # Batched labels (note: despite only one instance of class 1, each batch contains an instance)
        >>> print(sampler[0][1], sampler[1][1])
        [[0] [1] [0] [0] [0] [0]] [[1] [0] [0] [0] [0]]

        >>> # Batched weights (note: as a result of duplication, the weight of the class 1 instance was halved)
        >>> print(sampler[0][2], sampler[1][2])
        [[0.1] [0.05] [0.1] [0.1] [0.1] [0.1]] [[0.05] [0.1] [0.1] [0.1] [0.1]]

        >>> print(len(sampler))
        2

    Below is an example where a class with two members is copied to be distributed over 3 batches.
    We expect the two members to be duplicated to ensure at least member can be distributed to each
    batch. In this case, 2 batches will have 1 instance of the rare class, and 1 batch will have
    2 instances of the rare class.

    Example:

    .. code-block:: python

        >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1,1)
        >>> labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1]).reshape(-1,1)

        >>> sampler = DatasetWithBatching(data, labels, num_batches=3)

        >>> # Batched data
        >>> print(sampler[0][0], sampler[1][0], sampler[2][0])
        [[2] [6] [9] [0]] [[8] [1] [4] [7]] [[8] [5] [9] [3]]

        >>> # Batched labels (note: despite only two instance of class 1, the batches combined contain 4 instances)
        >>> print(sampler[0][1], sampler[1][1], sampler[2][1])
        [[0] [0] [1] [0]] [[1] [0] [0] [0]] [[1] [0] [1] [0]]

        >>> # Batched weights (note: as a result of duplication, the weight of the class 1 instances was halved)
        >>> print(sampler[0][2], sampler[1][2], sampler[2][2])
        [[0.1] [0.1] [0.05] [0.1]] [[0.05] [0.1] [0.1] [0.1]] [[0.05] [0.1] [0.05] [0.1]]

        >>> print(len(sampler))
        3

    Below is an example with 3 different classes, where two of the classes are rare, requiring
    each of the rare classes to be duplicated.

    Example:

    .. code-block:: python

        >>> data = np.arange(20).reshape(-1,1)
        >>> labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2]).reshape(-1,1)

        >>> sampler = DatasetWithBatching(data, labels, num_batches=3)

        >>> # Batched data
        >>> print(sampler[0][0], sampler[1][0], sampler[2][0])
        [[14] [7] [9] [18] [19] [4] [2] [3]]
         [[19] [17] [6] [0] [15] [5] [16] [10]]
         [[13] [12] [17] [8] [19] [18] [1] [11]]

        >>> # Batched labels (note: despite only two instance of class 1, the batches combined contain 4 instances)
        >>> print(sampler[0][1], sampler[1][1], sampler[2][1])
        [[0] [0] [0] [1] [2] [0] [0] [0]]
         [[2] [1] [0] [0] [0] [0] [0] [0]]
         [[0] [0] [1] [0] [2] [1] [0] [0]]

        >>> # Batched weights (note: as a result of duplication, the weight of the class 1 instance was halved)
        >>> print(sampler[0][2], sampler[1][2], sampler[2][2])
        [[0.05] [0.05] [0.05] [0.025] [0.017] [0.05] [0.05] [0.05]]
         [[0.017] [0.025] [0.05] [0.05] [0.05] [0.05] [0.05] [0.05]]
         [[0.05] [0.05] [0.025] [0.05] [0.017] [0.025] [0.05] [0.05]]

        >>> print(len(sampler))
        3

    """
    def __init__(self,
                 x_set,
                 y_set,
                 sample_weights=None,
                 batch_size=64,
                 num_batches=None,
                 seed=0,
                 shuffle=True,
                 **kwargs
                 ) -> None:
        super(DatasetWithBatching, self).__init__(
            x_set,
            y_set,
            sample_weights=sample_weights,
            batch_size=batch_size,
            num_batches=num_batches,
            seed=seed,
            shuffle=shuffle,
            mode=ModelType.CLASSIFICATION,
            **kwargs
        )
import numpy as np
import tensorflow as tf
from math import ceil
from imbal.util.backend.constants import ModelType

class DatasetWithBatching(tf.keras.utils.PyDataset):
    """
    An extension of `TensorFlow's PyDataset class <https://www.tensorflow.org/api_docs/python/tf/keras/utils/PyDataset>`_.
    This class can be used to ensure that data remains stratified during the
    batching process commonly used for training. This batch stratification
    is achieved by a number of techniques.

    Firstly, in the case of classification data, data-label pairs are checked to
    ensure that there is a sufficient number of samples of each class such that
    every batch contains at least one instance of each class. In the event
    that a particular class does not contain enough samples for this to be true,
    the samples in that class will be duplicated until a sufficient number of
    samples is reached. Then, the sample weights of each sample in the class is
    adjusted to account for the duplication, such that the sum of the weights of
    all copies of a particular sample is equal to the weight of the original
    singular sample.

    Once all classes have the property above, samples are distributed across each by via
    rotation among the batches, such that every batch can
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
        mode: Optional, default :code:`'classification'`. Should be set to :code:`'classification'` when working with
            discrete labels (classes), and :code:`'regression'` when working with continuous labels, as the stratification
            process differs depending on the label type.
        sort: Optional, default :code:`'descending'`. Used only in :code:`'regression'` mode, determines how
            the data will be sorted for stratification. In cases where larger data labels are rarer, this should
            be left as :code:`'descending'`. In cases where smaller data labels are rarer, this should set to
            :code:`'ascending'`.

    In :code:`classification` mode, data is stratified by class, ensuring that data is spread
    as evenly as possible across batches, such that at least 1 instance of a data point
    from each class is in each batch (making copies of class samples when necessary).
    In :code:`regression` mode, there are no explict classes to stratify data on. Instead,
    the data is sorted based on its label, then seperated into
    pseudo-classes of size equal to the number of batches. This means for each batch, the
    elements of data that are of similar size or ordering are guarenteed to be split
    across batches, leading to a more even data spread across batches.

    Each batch in the :code:`StratifiedBatcher` is stored as a tuple of the form
    :code:`(batch_x, batch_y, batch_weights)`. In this format, batches can be
    retrieved then manually fed to TensorFlow's :code:`model.fit()` or :code:`model.predict()`,
    but TensorFlow also allows for children of the :code:`PyDataset` class to be
    passed to its models as well.

    After instansiation, the number of batches generated can be retrieved by
    calling the :code:`len()` function on the :code:`StratifiedBatcher` object,
    as shown in the example below. Additionally, batches can be retrieved manually
    simply by indexing the object, such as :code:`sampler[i]`.

    In the example that follows, we see a dataset of 10 unique data points, labels 0 to 9.
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
        [6, 9, 7, 0, 5, 8] [9, 1, 2, 3, 4]

        >>> # Batched labels (note: despite only one instance of class 1, each batch contains an instance)
        >>> print(sampler[0][1], sampler[1][1])
        [0, 1, 0, 0, 0, 0] [1, 0, 0, 0, 0]

        >>> # Batched weights (note: as a result of duplication, the weight of the class 1 instance was halved)
        >>> print(sampler[0][2], sampler[1][2])
        [0.1, 0.05, 0.1, 0.1, 0.1, 0.1] [0.05, 0.1, 0.1, 0.1, 0.1]

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
        [2, 6, 9, 0] [8, 1, 4, 7] [8, 5, 9, 3]

        >>> # Batched labels (note: despite only two instance of class 1, the batches combined contain 4 instances)
        >>> print(sampler[0][1], sampler[1][1], sampler[2][1])
        [0, 0, 1, 0] [1, 0, 0, 0] [1, 0, 1, 0]

        >>> # Batched weights (note: as a result of duplication, the weight of the class 1 instances was halved)
        >>> print(sampler[0][2], sampler[1][2], sampler[2][2])
        [0.1, 0.1, 0.05, 0.1] [0.05, 0.1, 0.1, 0.1] [0.05, 0.1, 0.05, 0.1]

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
        [14, 7, 9, 18, 19, 4, 2, 3] [19, 17, 6, 0, 15, 5, 16, 10] [13, 12, 17, 8, 19, 18, 1, 11]

        >>> # Batched labels (note: despite only two instance of class 1, the batches combined contain 4 instances)
        >>> print(sampler[0][1], sampler[1][1], sampler[2][1])
        [0, 0, 0, 1, 2, 0, 0, 0] [2, 1, 0, 0, 0, 0, 0, 0] [0, 0, 1, 0, 2, 1, 0, 0]

        >>> # Batched weights (note: as a result of duplication, the weight of the class 1 instance was halved)
        >>> print(sampler[0][2], sampler[1][2], sampler[2][2])
        [0.05, 0.05, 0.05, 0.025, 0.017, 0.05, 0.05, 0.05] [0.017, 0.025, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05] [0.05, 0.05, 0.025, 0.05, 0.017, 0.025, 0.05, 0.05]

        >>> print(len(sampler))
        3

    """

    def _rebuild_batchable(self):
        self._batchable_data = np.concatenate(self._data_by_class)
        self._batchable_labels = np.concatenate(self._data_labels)
        self._batchable_weights = np.concatenate(self._data_weights)

    def __init__(self,
        x_set,
        y_set,
        sample_weights=None,
        batch_size=64,
        num_batches=None,
        seed=0,
        shuffle=True,
        mode=ModelType.CLASSIFICATION,
        sort='descending',
        **kwargs
    ) -> None:
        super(DatasetWithBatching, self).__init__(**kwargs)
        # Declare sampler attributes
        self._x_set = x_set
        self._y_set = y_set
        self._sample_weights = None
        self._seed = seed
        self._data_by_class = []
        self._data_labels = []
        self._data_weights = []
        self._shuffle = shuffle
        self._mode = mode
        self._sort = sort


        self._num_samples = len(self._x_set)

        output_labels = self._y_set
        if output_labels.ndim == 1:
            self._comp_values = output_labels
        elif output_labels.ndim == 2:
            if output_labels.shape[1] > 1:
                self._comp_values = np.argmax(output_labels, axis=1).astype(np.int32)
            else:
                self._comp_values = np.reshape(output_labels, (-1))
        else:
            raise ValueError('labels must be scalar values, or one-hot vectors')

        # Make sure num_batches is set (num_batches is easier to work with than batch_size)
        if num_batches is None:
            # Compute num_batches from batch_size and dataset size
            self._num_batches = int(np.ceil(self._num_samples / batch_size))
        else:
            self._num_batches = num_batches

        if sample_weights is None:
            self._sample_weights = np.ones([self._num_samples])
        else:
            self._sample_weights = np.reshape(sample_weights, (-1,))

        if not (self._num_samples == self._comp_values.shape[0] and self._num_samples == self._sample_weights.shape[0]):
            raise ValueError("Number of entries in data, labels, and weights must be equal")

        if self._mode == ModelType.REGRESSION:
            sort_order = np.argsort(self._comp_values)
            if self._sort == 'descending':
                sort_order = sort_order[::-1]

            self._x_set = self._x_set[sort_order]

            self._y_set = self._y_set[sort_order]
            unique_counts = [self._num_batches] * (self._y_set.shape[0] // self._num_batches) + [self._y_set.shape[0] % self._num_batches]

            self._sample_weights = self._sample_weights[sort_order]


            unique_classes = [1] * len(unique_counts)
        else:
            # Get a list of all labels in data, along with how many of each label
            unique_classes, unique_counts = np.unique(self._comp_values, return_counts=True)

        for idx, (label, count) in enumerate(zip(unique_classes, unique_counts)):
            duplicate_factor = int(np.ceil(self._num_batches / count)) if self._mode == ModelType.CLASSIFICATION else 1
            label_mask = self._comp_values == label
            if self._mode == ModelType.REGRESSION:
                class_data = self._x_set[idx*self._num_batches:idx*self._num_batches+count]
                class_samples = len(class_data)
                class_weights = self._sample_weights[idx*self._num_batches:idx*self._num_batches+count] / duplicate_factor
            else:
                class_data = self._x_set[label_mask]
                class_samples = len(class_data)
                class_weights = self._sample_weights[label_mask] / duplicate_factor
            if self._shuffle:
                rng = np.random.default_rng(self._seed + idx)
                shuffle_indices = rng.permutation(class_samples)
            else:
                shuffle_indices = np.arange(class_samples)

            class_data = class_data[shuffle_indices]

            class_weights = class_weights[shuffle_indices]

            self._data_by_class.append(np.tile(class_data, [duplicate_factor] + [1] * (self._x_set.ndim - 1)))
            self._data_weights.append(np.tile(class_weights, [duplicate_factor]))

            if self._mode == ModelType.REGRESSION:
                class_labels = self._y_set[idx*self._num_batches:idx*self._num_batches+count]
                class_labels = class_labels[shuffle_indices]
                self._data_labels.append(np.tile(class_labels, [duplicate_factor]))
            else:
                class_labels = np.tile([label], [count*duplicate_factor] + [1] * (self._y_set.ndim - 1))
                self._data_labels.append(class_labels)

        self._seed += self._num_batches

        self._rebuild_batchable()

        self._num_samples = len(self._batchable_data)

    @property
    def num_batches(self) -> int:
        return self._num_batches

    def __len__(self) -> int:
        return self._num_batches

    def __getitem__(self, idx: int) -> tuple:
        if idx < 0 or idx >= self._num_batches:
            raise IndexError('Index out of range')

        labels = self._batchable_labels[idx::self._num_batches]

        data = self._batchable_data[idx::self._num_batches]

        return (data,
            labels,
            np.reshape(self._batchable_weights[idx::self._num_batches], (-1, 1)))

    def on_epoch_end(self) -> None:
        """
        If :code:`shuffle` is True, batches are shuffled. Otherwise, does nothing.

        Returns:
            None
        """
        if not self._shuffle:
            return
        for i in range(len(self._data_by_class)):
            rng = np.random.default_rng(self._seed + i)
            indices = rng.permutation(len(self._data_by_class[i]))

            self._data_by_class[i] = self._data_by_class[i][indices]
            self._data_labels[i] = self._data_labels[i][indices]
            self._data_weights[i] = self._data_weights[i][indices]

        self._rebuild_batchable()

        self._seed += self._num_batches

    def unpack(self):
        return self._batchable_data, self._batchable_labels, self._batchable_weights
from imbal.util.backend.stratified_sampling.dataset_with_batching import DatasetWithBatching as UtilDataset
from imbal.util.backend.constants import ModelType

class DatasetWithBatching(UtilDataset):
    """
        An extension of `TensorFlow's PyDataset class <https://www.tensorflow.org/api_docs/python/tf/keras/utils/PyDataset>`_.
        This class can be used to ensure that data remains stratified during the
        batching process commonly used for training. This batch stratification
        is achieved by a number of techniques.

        In the case of regression, there are no explict classes to stratify data on. Instead,
        the data is sorted based on its label, then seperated into
        pseudo-classes of size equal to the number of batches. This means for each batch, the
        elements of data that are of similar size or ordering are guarenteed to be split
        across batches, leading to a more even data spread across batches.

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
            sort: Optional, default :code:`'descending'`. Determines how
                the data will be sorted for stratification. In cases where larger data labels are rarer, this should
                be left as :code:`'descending'`. In cases where smaller data labels are rarer, this should set to
                :code:`'ascending'`.

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
        The label for each data point is also 0 to 9.

        The example calls for the creation of two batches. Therefore, in order to maintain
        that data whose labels are close in value are stratified into separate batches,
        we should expect that labels 0 and 1 are in separate batches, 2 and 3, 4 and 5, and so on.

        Example:

        .. code-block:: python

            >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1,1)
            >>> labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1,1)

            >>> sampler = DatasetWithBatching(data, labels, num_batches=2)

            >>> # Batched data
            >>> print(sampler[0][0], sampler[1][0])
            [[4] [7] [3] [1] [9]] [[0] [2] [8] [6] [5]]

            >>> # Label correspondences are preserved, and the stratified property is present
            >>> print(sampler[0][1], sampler[1][1])
            [[4] [7] [3] [1] [9]] [[0] [2] [8] [6] [5]]

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

         """
    def __init__(self,
                 x_set,
                 y_set,
                 sample_weights=None,
                 batch_size=64,
                 num_batches=None,
                 seed=0,
                 shuffle=True,
                 sort='descending',
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
            mode=ModelType.REGRESSION,
            sort=sort,
            **kwargs
        )
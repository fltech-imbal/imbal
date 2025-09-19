import tensorflow as tf

class SimpleDataset(tf.keras.utils.PyDataset):
    """
    A simple extension of the `tf.keras.utils.PyDataset <https://www.tensorflow.org/api_docs/python/tf/keras/utils/PyDataset>`_
    class that allows for the storage of data, labels, and (optionally) weights.
    """
    def __init__(self,
        x_set,
        y_set,
        sample_weights=None,
        **kwargs
    ) -> None:
        super(SimpleDataset, self).__init__(**kwargs)

        self._x_set = x_set
        self._y_set = y_set
        self._sample_weights = sample_weights

        if self._x_set.shape[0] != self._y_set.shape[0]:
            raise ValueError('x_set and y_set must have the same sized dimension 0')
        if sample_weights is None:
            self._data_tuples = list(zip(self._x_set, self._y_set))
        else:
            if self._x_set.shape[0] != self._sample_weights.shape[0]:
                raise ValueError('x_set, y_set and sample_weights must have the same sized dimension 0')
            self._data_tuples = list(zip(self._x_set, self._y_set, self._sample_weights))

    def set_weights(self, sample_weights):
        """
        Sets the weights of the samples in the dataset.

        Args:
            sample_weights: A NumPy array of shape [num_samples, 1], specifying
                a sample weight for each sample in the dataset.

        Returns:
            None
        """
        self._sample_weights = sample_weights
        if sample_weights is None:
            self._data_tuples = list(zip(self._x_set, self._y_set))
        else:
            if self._x_set.shape[0] != self._sample_weights.shape[0]:
                raise ValueError('x_set, y_set and sample_weights must have the same sized dimension 0')
            self._data_tuples = list(zip(self._x_set, self._y_set, self._sample_weights))

    def get_data(self):
        """
        Returns the data column of the dataset.

        Returns:
            A NumPy array containing all data samples.
        """
        return self._x_set
    def get_labels(self):
        """
        Returns the labels column of the dataset.

        Returns:
            A NumPy array containing all labels.
        """
        return self._y_set
    def get_weights(self):
        """
        Returns the weights column of the dataset, or :code:`None` if no weights are stored.

        Returns:
            A NumPy array containing all weights.
        """
        return self._sample_weights
    def get_unzipped(self):
        """
        Returns the data-label-weight tuples stored in the dataset
        in an unzipped format (i.e., this function returns the
        equivalent of :code:`(dataset.get_data(), dataset.get_labels(), dataset.get_weights())`)

        Returns:
            A tuple of the form :code:`(data, labels, weights)`, or
            :code:`(data, labels)` if :code:`sample_weights` is :code:`None`
        """
        if self._sample_weights is None:
            return self._x_set, self._y_set
        else:
            return self._x_set, self._y_set, self._sample_weights

    def __len__(self) -> int:
        return len(self._data_tuples)

    def __getitem__(self, idx: int) -> tuple:
        return self._data_tuples[idx]
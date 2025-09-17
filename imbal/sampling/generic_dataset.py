import tensorflow as tf
from numpy.typing import NDArray

class GenericDataset(tf.keras.utils.PyDataset):
    def __init__(self,
        x_set,
        y_set,
        sample_weights=None,
        **kwargs
    ) -> None:
        super(GenericDataset, self).__init__(**kwargs)

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

    def set_weights(self, sample_weights) -> None:
        self._sample_weights = sample_weights
        if sample_weights is None:
            self._data_tuples = list(zip(self._x_set, self._y_set))
        else:
            if self._x_set.shape[0] != self._sample_weights.shape[0]:
                raise ValueError('x_set, y_set and sample_weights must have the same sized dimension 0')
            self._data_tuples = list(zip(self._x_set, self._y_set, self._sample_weights))

    def get_data(self) -> NDArray:
        return self._x_set
    def get_labels(self) -> NDArray:
        return self._y_set
    def get_weights(self) -> NDArray:
        return self._sample_weights
    def get_unzipped(self) -> tuple:
        return self._x_set, self._y_set, self._sample_weights

    def __len__(self) -> int:
        return len(self._data_tuples)

    def __getitem__(self, idx: int) -> tuple:
        return self._data_tuples[idx]
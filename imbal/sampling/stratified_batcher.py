import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from math import ceil

class StratifiedBatcher(tf.keras.utils.PyDataset):
    def __init__(self,
        x_set,
        y_set,
        sample_weights=None,
        batch_size=64,
        num_batches=None,
        dtype=None,
        seed=0,
        shuffle=True,
        mode='class',
        **kwargs
    ) -> None:
        super(StratifiedBatcher, self).__init__(**kwargs)
        # Declare sampler attributes
        self._x_set : Tensor = tf.constant(x_set, dtype=dtype)
        self._y_set : Tensor = tf.reshape(tf.constant(y_set, dtype=dtype), (-1,))
        self._sample_weights = None
        self._seed = seed
        self._data_by_class = []
        self._data_labels = []
        self._data_weights = []
        self._weight_sum = None
        self._mode = mode

        # Make sure num_batches is set (num_batches is easier to work with than batch_size)
        if num_batches is None:
            # Compute num_batches from batch_size and dataset size
            self._num_batches = int(np.ceil(x_set.shape[0] / batch_size))
        else:
            self._num_batches = num_batches

        if sample_weights is None:
            self._sample_weights = np.ones([x_set.shape[0]]) / x_set.shape[0]
        else:
            self._sample_weights = tf.reshape(tf.constant(sample_weights, dtype=dtype), (-1,))

        if not (self._x_set.shape[0] == self._y_set.shape[0] and self._x_set.shape[0] == self._sample_weights.shape[0]):
            raise ValueError("Number of entries in data, labels, and weights must be equal")

        self._weight_sum = float(sum(self._sample_weights))

        unique_classes = None
        unique_counts = None
        if self._mode == 'reg':
            unique_counts = [self._num_batches] * (self._y_set.shape[0] // self._num_batches) + [self._y_set.shape[0] % self._num_batches]
            unique_classes = [1] * len(unique_counts)
        else:
            # Get a list of all labels in data, along with how many of each label
            unique_classes, _, unique_counts = tf.unique_with_counts(self._y_set)
            unique_classes, unique_counts = unique_classes.numpy(), unique_counts.numpy()

        for idx, (label, count) in enumerate(zip(unique_classes, unique_counts)):
            duplicate_factor = int(np.ceil(self._num_batches / count)) if mode == 'class' else 1
            class_data = None
            class_weights = None

            if self._mode == 'reg':
                class_data = self._x_set[idx*self._num_batches:idx*self._num_batches+count]
                class_weights = self._sample_weights[idx*self._num_batches:idx*self._num_batches+count] / duplicate_factor
            else:
                class_data = tf.boolean_mask(self._x_set, self._y_set == label, axis=0)
                class_weights = tf.boolean_mask(self._sample_weights, self._y_set == label, axis=0) / duplicate_factor

            indices = tf.random.experimental.stateless_shuffle(tf.range(class_data.shape[0]),
                                                               seed=[self._seed + idx, self._seed + idx])
            class_data = tf.gather(class_data, indices)
            class_weights = tf.gather(class_weights, indices)

            self._data_by_class.append(tf.tile(class_data, tf.constant([duplicate_factor] + [1] * (self._x_set.ndim - 1), dtype=tf.int32)))
            self._data_weights.append(tf.tile(class_weights, tf.constant([duplicate_factor])))

            if self._mode == 'reg':
                class_labels = self._y_set[idx*self._num_batches:idx*self._num_batches+count]
                class_labels = tf.gather(class_labels, indices)
                self._data_labels.append(tf.tile(class_labels, tf.constant([duplicate_factor])))
            else:
                self._data_labels.append(tf.tile(tf.fill([count], label), tf.constant([duplicate_factor])))

        self._seed += self._num_batches

        self._batchable_data = tf.concat(self._data_by_class, 0)
        self._batchable_labels = tf.concat(self._data_labels, 0)
        self._batchable_weights = tf.concat(self._data_weights, 0)

    def __len__(self) -> int:
        return self._num_batches

    def __getitem__(self, idx: int) -> tuple:
        if idx < 0 or idx >= self._num_batches:
            raise IndexError('Index out of range')

        batch_size = ceil((self._batchable_data.shape[0] - idx) / self._num_batches)

        indices = tf.random.experimental.stateless_shuffle(tf.range(batch_size),
                                                           seed=[self._seed + idx, self._seed + idx])

        return (tf.gather(self._batchable_data[idx::self._num_batches], indices),
            tf.reshape(tf.gather(self._batchable_labels[idx::self._num_batches], indices), (-1, 1)),
            tf.reshape(tf.gather(self._batchable_weights[idx::self._num_batches], indices), (-1, 1)))

    def on_epoch_end(self) -> None:
        for i in range(len(self._data_by_class)):
            indices = tf.random.experimental.stateless_shuffle(tf.range(len(self._data_by_class[i])),
                                                     seed=[self._seed + i, self._seed + i])
            self._data_by_class[i] = tf.gather(self._data_by_class[i], indices)
            self._data_weights[i] = tf.gather(self._data_weights[i], indices)
            if self._mode == 'reg':
                self._data_labels[i] = tf.gather(self._data_labels[i], indices)

        self._batchable_data = tf.concat(self._data_by_class, 0)
        self._batchable_weights = tf.concat(self._data_weights, 0)
        if self._mode == 'reg':
            self._batchable_labels = tf.concat(self._data_labels, 0)
        self._seed += self._num_batches
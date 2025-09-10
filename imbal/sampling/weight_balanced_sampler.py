import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from typing import Any
from math import ceil

class WeightBalancedSampler(tf.keras.utils.PyDataset):
    def __init__(self, x_set, y_set,
                 batch_size=64,
                 num_batches=None,
                 dtype=None,
                 sample_weights=None,
                 class_weights=None,
                 seed=0,
                 **kwargs) -> None:
        super(WeightBalancedSampler, self).__init__(**kwargs)
        # Declare sampler attributes
        self._x_set : Tensor = tf.constant(x_set, dtype=dtype)
        self._y_set : Tensor = tf.reshape(tf.constant(y_set, dtype=dtype), (-1,))
        self._sample_weights : dict[Any, float] = sample_weights
        self._class_weights : dict[Any, float] = class_weights
        self._seed = seed
        self._data_by_class = []
        self._data_labels = []
        self._data_weights = []
        self._weight_sum = None

        # Make sure num_batches is set (num_batches is easier to work with than batch_size)
        if num_batches is None:
            # Compute num_batches from batch_size and dataset size
            self._num_batches = int(np.ceil(x_set.shape[0] / batch_size))
        else:
            self._num_batches = num_batches

        # Get a list of all labels in data, along with how many of each label
        unique_classes, _, unique_counts = tf.unique_with_counts(self._y_set)
        unique_classes, unique_counts = unique_classes.numpy(), unique_counts.numpy()

        if self._sample_weights is not None:
            self._class_weights = {}
            for label, count in zip(unique_classes, unique_counts):
                if label in self._sample_weights:
                    self._class_weights.update({ label: count * self._sample_weights[label] })
                else:
                    self._class_weights.update({ label: count })

        if self._class_weights is None:
            self._class_weights = { label: 1 for label in unique_classes }
        else:
            for label, count in zip(unique_classes, unique_counts):
                if label not in self._class_weights:
                    self._class_weights[label] = 1

        self._weight_sum = sum(self._class_weights[label] for label in self._class_weights)

        for idx, (label, count) in enumerate(zip(unique_classes, unique_counts)):
            duplicate_factor = int(np.ceil(self._num_batches / count))
            self._class_weights[label] /= duplicate_factor
            self._data_by_class.append(tf.random.shuffle(
                tf.boolean_mask(
                    tf.tile(self._x_set, tf.constant([duplicate_factor] + [1 for i in range(self._x_set.ndim - 1)], dtype=tf.int32)),
                    tf.tile(self._y_set == label, tf.constant([duplicate_factor], dtype=tf.int32)),
                    axis=0
                ),
                seed=self._seed + idx
            ))
            self._data_labels.append(tf.tile(tf.fill([count], label), tf.constant([duplicate_factor])))
            self._data_weights.append(tf.tile(tf.fill([count], self._class_weights[label] / count), tf.constant([duplicate_factor])) / self._weight_sum)

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
            self._data_by_class[i] = tf.reshape(tf.random.shuffle(
                tf.reshape(self._data_by_class[i], (-1,)),
                seed=self._seed + i
            ), (-1, 1))

        self._batchable_data = tf.concat(self._data_by_class, 0)
        self._seed += self._num_batches
    
    # @classmethod
    # def split_stratify(cls, x_set, y_set,
    #              batch_size=64,
    #              num_batches=None,
    #              dtype=None,
    #              sample_weights=None,
    #              class_weights=None,
    #              training_split = 0.8,
    #              test_split = None,
    #              seed=0,
    #              **kwargs):
    #
    #     _x_set: Tensor = tf.constant(x_set, dtype=dtype)
    #     _y_set: Tensor = tf.reshape(tf.constant(y_set, dtype=dtype), (-1,))
    #
    #     unique_classes, _, count = tf.unique_with_counts(_y_set)
    #
    #     if sample_weights is not None:
    #         class_weights = {}
    #         for label, label_count in zip(unique_classes, count):
    #             if label in sample_weights:
    #                 class_weights.update({ label: label_count * sample_weights[label] })
    #             else:
    #                 class_weights.update({ label: label_count })
    #
    #     if class_weights is None:
    #         class_weights = { label: label_count for label, label_count in zip(unique_classes, count) }
    #     else:
    #         for label in unique_classes:
    #             if label not in class_weights:
    #                 class_weights[label] = 1
    #
    #     if test_split is not None:
    #         training_split = 1 - test_split
    #
    #     train_data = []
    #     train_labels = []
    #     train_weights = []
    #     test_data = []
    #     test_labels = []
    #     test_weights = []
    #
    #     for idx, label in enumerate(unique_classes):
    #         data_for_class = tf.boolean_mask(
    #                                 _x_set,
    #                                 _y_set == label,
    #                                 axis=0
    #                             )
    #         labels_for_class = tf.fill([count[idx]], label)
    #         weights_for_class = tf.fill([count[idx]], class_weights[label] / count[idx])
    #
    #         split_point = round(len(data_for_class) * training_split)
    #
    #         train_data.append(data_for_class[:split_point])
    #         train_labels.append(labels_for_class[:split_point])
    #         train_weights.append(weights_for_class[:split_point])
    #
    #         test_data.append(data_for_class[split_point:])
    #         test_labels.append(labels_for_class[split_point:])
    #         test_weights.append(weights_for_class[split_point:])
    #
    #         return (
    #             StratifiedSampler(
    #                 train_data,
    #                 train_labels,
    #                 sample_weights=train_weights,
    #                 class_weights=class_weights,
    #                 seed=seed,
    #                 dtype=dtype,
    #                 num_batches=num_batches,
    #                 batch_size=batch_size,
    #                 **kwargs
    #             ),
    #             StratifiedSampler(
    #                 test_data,
    #                 test_labels,
    #                 sample_weights=test_weights,
    #                 class_weights=class_weights,
    #                 seed=seed,
    #                 dtype=dtype,
    #                 num_batches=num_batches,
    #                 batch_size=batch_size,
    #                 **kwargs
    #             )
    #         )


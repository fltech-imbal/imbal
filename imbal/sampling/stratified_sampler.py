import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from typing import Any

class StratifiedSampler(tf.keras.utils.PyDataset):
    def __init__(self, x_set, y_set,
                 batch_size=64,
                 num_batches=None,
                 dtype=None,
                 sample_weights=None,
                 class_weights=None,
                 seed=0,
                 **kwargs) -> None:
        super(StratifiedSampler, self).__init__(**kwargs)
        self._x_set : Tensor = tf.constant(x_set, dtype=dtype)
        self._y_set : Tensor = tf.reshape(tf.constant(y_set, dtype=dtype), (-1,))
        self._sample_weights : dict[Any, float] = sample_weights
        self._class_weights : dict[Any, float] = class_weights
        self._seed = seed
        self._built = False

        self._stratified_data = None
        self._stratified_labels = None
        self._stratified_weights = None

        if num_batches is None:
            self._num_batches = int(np.ceil(x_set.shape[0] / batch_size))
        else:
            self._num_batches = num_batches

    def build(self) -> None:
        unique_classes, _, count = tf.unique_with_counts(self._y_set)
        unique_classes = unique_classes.numpy()
        count = count.numpy()

        if self._sample_weights is not None:
            self._class_weights = {}
            for label, label_count in zip(unique_classes, count):
                if label in self._sample_weights:
                    self._class_weights.update({ label: label_count * self._sample_weights[label] })
                else:
                    self._class_weights.update({ label: label_count })

        if self._class_weights is None:
            self._class_weights = { label: label_count for label, label_count in zip(unique_classes, count) }
        else:
            for label in unique_classes:
                if label not in self._class_weights:
                    self._class_weights[label] = 1

        weight_sum = sum(self._class_weights[label] for label in self._class_weights)

        data_by_class = []
        labels = []
        weights = []

        for idx, label in enumerate(unique_classes):
            duplicate_factor = int(np.ceil(self._num_batches / count[idx]))
            self._class_weights[label] = self._class_weights[label] / duplicate_factor

            data_by_class.append(tf.random.shuffle(
                tf.boolean_mask(
                    tf.tile(self._x_set, tf.constant([duplicate_factor] + [1 for i in range(self._x_set.ndim - 1)], dtype=tf.int32)),
                    tf.tile(self._y_set == label, tf.constant([duplicate_factor], dtype=tf.int32)),
                    axis=0
                ),
                seed=self._seed + idx
            ))
            labels.append(tf.tile(tf.fill([count[idx]], label), tf.constant([duplicate_factor])))
            weights.append(tf.tile(tf.fill([count[idx]], self._class_weights[label] / count[idx]), tf.constant([duplicate_factor])))

        self._stratified_data = tf.concat(data_by_class, 0)
        self._stratified_labels = tf.concat(labels, 0)
        self._stratified_weights = tf.concat(weights, 0) / weight_sum

        self._built = True


    def __len__(self) -> int:
        return self._num_batches

    def __getitem__(self, idx: int) -> tuple:
        if not self._built:
            raise RuntimeError('StratifiedSampler must be built before batches can be retrieved')
        if idx < 0 or idx >= self._num_batches:
            raise IndexError('Index out of range')

        batch_data = self._stratified_data[idx::self._num_batches]
        batch_labels = self._stratified_labels[idx::self._num_batches]
        batch_weights = self._stratified_weights[idx::self._num_batches]

        indices = tf.random.experimental.stateless_shuffle(tf.range(batch_data.shape[0]), seed=[self._seed + idx, self._seed + idx])

        return (tf.gather(batch_data, indices),
                tf.reshape(tf.gather(batch_labels, indices), (-1, 1)),
                tf.reshape(tf.gather(batch_weights, indices), (-1, 1)))
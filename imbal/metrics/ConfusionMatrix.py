from typing_extensions import Self
from typing import List
from numpy.typing import NDArray
import tensorflow as tf
from keras.src import ops

class ConfusionMatrix:
    def __init__(self, actual : NDArray,
                 predicted : NDArray,
                 threshold: float | None = None) -> None:

        actual = tf.reshape(actual, (-1,))
        predicted = tf.reshape(predicted, (-1,))
        assert actual.shape == predicted.shape

        if threshold is not None:
            predicted = tf.greater(ops.cast(predicted, tf.float32), threshold)

        # Convert inputs to tensors immediately (prevent slowdown from multiple conversions)
        actual_tensor = ops.cast(ops.convert_to_tensor(actual), tf.bool)
        predicted_tensor = ops.cast(ops.convert_to_tensor(predicted), tf.bool)

        self._sample_size = tf.cast(tf.size(actual), tf.int64)

        predicted_positives = tf.math.count_nonzero(predicted_tensor)
        actual_positives = tf.math.count_nonzero(actual_tensor)

        self._true_positives = tf.math.count_nonzero(tf.logical_and(actual_tensor, predicted_tensor))
        self._false_positives = predicted_positives - self._true_positives
        self._false_negatives = actual_positives - self._true_positives
        self._true_negatives =  self._sample_size - self._true_positives - self._false_positives - self._false_negatives

    def tp(self) -> int:
        return self._true_positives
    def fp(self) -> int:
        return self._false_positives
    def tn(self) -> int:
        return self._true_negatives
    def fn(self) -> int:
        return self._false_negatives
    def pos(self) -> int:
        return self._true_positives + self._false_negatives
    def neg(self) -> int:
        return self._true_negatives + self._false_positives
    def ppos(self) -> int:
        return self._true_positives + self._false_positives
    def pneg(self) -> int:
        return self._true_negatives + self._false_negatives
    def sample_size(self) -> int:
        return self._sample_size


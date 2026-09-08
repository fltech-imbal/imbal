from tensorflow import Tensor
from keras.src import ops
from numpy.typing import NDArray

from .confusion_matrix_metric import ConfusionMatrixMetric

def weighted_sum(
    val: Tensor,
    weights: NDArray
) -> Tensor:
    if weights is not None:
        val = ops.multiply(val, ops.expand_dims(weights, 1))
    return ops.sum(val, axis=0)
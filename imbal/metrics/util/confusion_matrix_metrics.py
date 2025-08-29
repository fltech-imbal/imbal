from numpy.typing import NDArray
from tensorflow import Tensor
from keras.src import ops

def tp(y_true : NDArray | Tensor, y_pred : NDArray | Tensor) -> NDArray | Tensor:
    return y_true * y_pred
def fp(y_true : NDArray | Tensor, y_pred : NDArray | Tensor) ->  NDArray | Tensor:
    return (1 - y_true) * y_pred
def tn(y_true : NDArray | Tensor, y_pred : NDArray | Tensor) ->  NDArray | Tensor:
    return (1 - y_true) * (1 - y_pred)
def fn(y_true : NDArray | Tensor, y_pred : NDArray | Tensor) ->  NDArray | Tensor:
    return y_true * (1 - y_pred)

def weighted_sum(
    val: Tensor,
    weights: NDArray
) -> Tensor:
    if weights is not None:
        val = ops.multiply(val, ops.expand_dims(weights, 1))
    return ops.sum(val, axis=0)


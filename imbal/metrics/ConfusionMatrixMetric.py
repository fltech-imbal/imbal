from keras import Variable
from numpy.typing import NDArray
from typing import Tuple
from keras.src.metrics.metric import Metric
from tensorflow import Tensor
from keras.src import ops, initializers

class ConfusionMatrixMetric(Metric):
    def __init__(
        self,
        threshold : None | float = 0.5,
        name : str = 'confusion_matrix_metric',
        dtype : type | None = None
    ) -> None:

        super().__init__(name=name, dtype=dtype)

        self._threshold = threshold
        self._built = False
        self._axis = 0

    def add_zeros_variable(self, name) -> Variable:
        return self.add_variable(
            name=name,
            shape=(1,),
            initializer=initializers.Zeros(),
            dtype=self.dtype,
        )

    def build(
        self,
        y_true_shape : Tuple,
        y_pred_shape : Tuple
    ) -> None:
        if len(y_pred_shape) != 2 or len(y_true_shape) != 2:
            raise ValueError(
                "Confusion matrix metrics expect 2D inputs with shape "
                "(batch_size, output_dim). Received input "
                f"shapes: y_pred.shape={y_pred_shape} and "
                f"y_true.shape={y_true_shape}."
            )
        if y_pred_shape[-1] is None or y_true_shape[-1] is None:
            raise ValueError(
                "Confusion matrix metrics expect 2D inputs with shape "
                "(batch_size, output_dim), with output_dim fully "
                "defined (not None). Received input "
                f"shapes: y_pred.shape={y_pred_shape} and "
                f"y_true.shape={y_true_shape}."
            )

        self._built = False

    def update_state(self,
         y_true : NDArray | Tensor,
         y_pred : NDArray | Tensor,
         sample_weight : NDArray | Tensor | None = None
    ) -> None:
        y_true = ops.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = ops.convert_to_tensor(y_pred, dtype=self.dtype)
        if not self._built:
            self.build(y_true.shape, y_pred.shape)
        if not self._built:
            raise NotImplementedError("Please ensure _build function is implemented properly.")

        if self._threshold is None:
            threshold = ops.max(y_pred, axis=-1, keepdims=True)
            y_pred = ops.logical_and(
                y_pred >= threshold, ops.abs(y_pred) > 1e-9
            )
        else:
            y_pred = y_pred > self._threshold

        y_pred = ops.cast(y_pred, dtype=self.dtype)
        y_true = ops.cast(y_true, dtype=self.dtype)
        if sample_weight is not None:
            sample_weight = ops.convert_to_tensor(
                sample_weight, dtype=self.dtype
            )

        self.complete_update(y_true, y_pred, sample_weight)

    def complete_update(
            self,
            y_true: NDArray | Tensor,
            y_pred: NDArray | Tensor,
            sample_weight: NDArray | Tensor | None = None
    ):
        raise NotImplementedError

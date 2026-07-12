from numpy.typing import NDArray
from tensorflow import Tensor, Variable
import tensorflow as tf
from enum import Enum
import keras
from imbal.metrics.util import weighted_sum

class ConfusionMatrixData(Enum):
    TRUE_POSITIVE = 'tp'
    TRUE_NEGATIVE = 'tn'
    FALSE_POSITIVE = 'fp'
    FALSE_NEGATIVE = 'fn'
    POSITIVE = 'pos'
    NEGATIVE = 'neg'
    PREDICTED_POSITIVE = 'ppos'
    PREDICTED_NEGATIVE = 'pneg'
    SAMPLE_SIZE = 'ss'

class ConfusionMatrix:

    @staticmethod
    def compute(
            variables_to_update : dict[str | ConfusionMatrixData, Variable],
            y_true: NDArray | Tensor,
            y_pred: NDArray | Tensor,
            sample_weight: NDArray | Tensor | None = None,
            dtype : type | None = None,
            mode : str = 'assign_add'
    ) -> None | dict:

        if variables_to_update is None:
            return {
                ConfusionMatrixData.TRUE_POSITIVE : weighted_sum(y_true * y_pred, sample_weight),
                ConfusionMatrixData.TRUE_NEGATIVE : weighted_sum((1 - y_true) * (1 - y_pred), sample_weight),
                ConfusionMatrixData.FALSE_POSITIVE : weighted_sum((1 - y_true) * y_pred, sample_weight),
                ConfusionMatrixData.FALSE_NEGATIVE : weighted_sum(y_true * (1 - y_pred), sample_weight),
                ConfusionMatrixData.POSITIVE : weighted_sum(y_true, sample_weight),
                ConfusionMatrixData.NEGATIVE : weighted_sum(1 - y_true, sample_weight),
                ConfusionMatrixData.PREDICTED_POSITIVE : weighted_sum(y_pred, sample_weight),
                ConfusionMatrixData.PREDICTED_NEGATIVE : weighted_sum(1 - y_pred, sample_weight),
                ConfusionMatrixData.SAMPLE_SIZE : weighted_sum(tf.ones(tf.shape(y_true), dtype=dtype), sample_weight)
            }

        if mode not in ['assign', 'assign_add']:
            raise ValueError('ConfusionMatrix mode must be either "assign" or "assign_add."'
                             f'Received: "{mode}"')

        if not any(
                key for key in variables_to_update if key in list(ConfusionMatrixData)
        ):
            raise ValueError(
                "Please provide at least one valid confusion matrix "
                "variable to update. Valid variable key options are: "
                f'"{list(ConfusionMatrixData)}". '
                f'Received: "{variables_to_update.keys()}"'
            )
        invalid_keys = [
            key for key in variables_to_update if key not in list(ConfusionMatrixData)
        ]
        if invalid_keys:
            raise ValueError(
                f'Invalid keys: "{invalid_keys}". '
                f'Valid variable key options are: "{list(ConfusionMatrixData)}"'
            )

        def update_variable(variable : Variable, value : Tensor) -> None:
            if mode == 'assign':
                variable.assign(value)
            else:
                variable.assign_add(value)

        for key in variables_to_update:
            match key:
                case 'tp' | ConfusionMatrixData.TRUE_POSITIVE:
                    update_variable(variables_to_update[key], weighted_sum(y_true * y_pred, sample_weight))
                case 'tn' | ConfusionMatrixData.TRUE_NEGATIVE:
                    update_variable(variables_to_update[key], weighted_sum((1 - y_true) * (1 - y_pred), sample_weight))
                case 'fp' | ConfusionMatrixData.FALSE_POSITIVE:
                    update_variable(variables_to_update[key], weighted_sum((1 - y_true) * y_pred, sample_weight))
                case 'fn' | ConfusionMatrixData.FALSE_NEGATIVE:
                    update_variable(variables_to_update[key], weighted_sum(y_true * (1 - y_pred), sample_weight))
                case 'pos' | ConfusionMatrixData.POSITIVE:
                    update_variable(variables_to_update[key], weighted_sum(y_true, sample_weight))
                case 'neg' | ConfusionMatrixData.NEGATIVE:
                    update_variable(variables_to_update[key], weighted_sum(1 - y_true, sample_weight))
                case 'ppos' | ConfusionMatrixData.PREDICTED_POSITIVE:
                    update_variable(variables_to_update[key], weighted_sum(y_pred, sample_weight))
                case 'pneg' | ConfusionMatrixData.PREDICTED_NEGATIVE:
                    update_variable(variables_to_update[key], weighted_sum(1 - y_pred, sample_weight))
                case 'ss' | ConfusionMatrixData.SAMPLE_SIZE:
                    update_variable(variables_to_update[key], weighted_sum(tf.ones(tf.shape(y_true), dtype=dtype), sample_weight))
        return None
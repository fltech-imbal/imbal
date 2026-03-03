import numpy as np
import warnings

def positive_model_layer_index(
    model,
    desired_index
):
    num_layers = len(model.layers)
    if desired_index < 0:
        desired_index = num_layers + desired_index

    if desired_index >= num_layers:
        raise ValueError("Desired layer index cannot be greater than or equal to number of layers, or less than -(number of layers)")

    return desired_index

def safe_object_unwrap(obj, obj_type):
    if isinstance(obj, obj_type):
        return obj.to_dict()
    elif obj is None:
        return {}
    else:
        return obj

def is_list_like(obj):
    return isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, np.ndarray)

def verify_weight_scale(weights, show_warning=True, axis=None):
    if weights is None:
        return weights
    weights = np.array(weights, dtype=np.float32)

    if axis is None:
        num_samples = weights.size
        current_sum = np.sum(weights)
        if abs(current_sum - num_samples) > 1e-3 * num_samples:
            if show_warning:
                warnings.warn("Weights do not sum to n. TensorFlow expects provided weights to sum to n. Weights will be rescaled.")
            weights *= num_samples / current_sum
    else:
        num_samples = weights.shape[axis]
        current_sum = np.sum(weights, axis=axis, keepdims=True)
        if np.any(np.abs(current_sum - num_samples) > 1e-3 * num_samples):
            if show_warning:
                warnings.warn("Weights do not sum to N along axis. TensorFlow expects provided weights to sum to n. Weights will be rescaled.")
            weights *= num_samples / current_sum

    return weights
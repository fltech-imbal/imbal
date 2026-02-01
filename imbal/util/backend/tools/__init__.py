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

def verify_weight_scale(weights, show_warning=True):
    if weights is not None and abs(np.sum(weights) - weights.shape[0]) > 10e-3:
        if show_warning:
            warnings.warn("Weights provided to fit function do not sum to n, where n is the number of "
                          "samples. TensorFlow expects provided weights to sum to n. Weights will be scaled.")
        weights *= weights.shape[0] / np.sum(weights)
    return weights
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
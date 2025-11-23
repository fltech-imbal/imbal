def get_last_trainable_index(
    model,
    n_to_last=1,
    desired_layer_index=-2,
):

    best_fit = None, None
    return_on_find = False
    num_layers = len(model.layers)
    if desired_layer_index < 0:
        desired_layer_index = num_layers + desired_layer_index

    if desired_layer_index >= num_layers:
        raise ValueError("Desired layer index cannot be greater than or equal to number of layers, or less than -(number of layers)")

    for i in range(num_layers):
        if i < n_to_last - 1:
            continue
        adjusted_index = num_layers - 1 - i
        current_layer = model.layers[adjusted_index]
        if hasattr(current_layer, 'kernel_initializer') and hasattr(current_layer, 'bias_initializer'):
            best_fit = current_layer, adjusted_index
            if return_on_find:
                return best_fit
        if desired_layer_index == adjusted_index:
            if best_fit[0] is None:
                return_on_find = True
            else:
                return best_fit

    return None, None
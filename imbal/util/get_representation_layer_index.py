import warnings

def get_representation_layer_index(
    model,
    desired_layer_index=-2,
):

    num_layers = len(model.layers)
    if desired_layer_index < 0:
        desired_layer_index = num_layers + desired_layer_index

    if desired_layer_index >= num_layers:
        raise ValueError("Desired layer index cannot be greater than or equal to number of layers, or less than -(number of layers)")

    best_fit = None

    for i in range(num_layers):
        adjusted_index = num_layers - 1 - i
        current_layer = model.layers[adjusted_index]
        if hasattr(current_layer, 'kernel_initializer') and hasattr(current_layer, 'bias_initializer'):
            if adjusted_index == 0:
                warnings.warn('No representation layer found. Model contains only one trainable layer.')
                return None, None
            else:
                best_fit =  model.layers[adjusted_index - 1], adjusted_index - 1
                break

    if best_fit is None:
        warnings.warn('No representation layer found. Model contains no trainable layers.')
        return None, None

    if desired_layer_index > best_fit[1]:
        warnings.warn('Specified representation layer is not followed by any other trainable layers. Overridden with '
                      f'layer {best_fit[1]}, which is followed by at least one trainable layer.')
    return best_fit
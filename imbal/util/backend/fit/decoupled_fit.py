import imbal.util as util
import numpy as np
import warnings
from math import ceil
from .balanced_fit import balanced_fit
from imbal.util.backend.tools import positive_model_layer_index, safe_object_unwrap

def decoupled_fit(
    model,
    x=None,
    y=None,
    compile_parameters=None,
    stage_one_compile_parameters=None,
    stage_two_compile_parameters=None,
    sample_weights=None,
    sample_densities=None,
    class_weights=None,
    batch_size=32,
    epochs=1,
    validation_data=None,
    shuffle=True,
    representation_layer_index=-2,
    stratify_batches=True,
    aed_for_representation=True,
    mode='classification'
):
    model.trainable = True

    compile_parameters = safe_object_unwrap(compile_parameters, util.ModelCompileParameters)

    if isinstance(stage_one_compile_parameters, util.ModelCompileParameters):
        stage_one_compile_parameters = stage_one_compile_parameters.to_dict()
    if isinstance(stage_two_compile_parameters, util.ModelCompileParameters):
        stage_two_compile_parameters = stage_two_compile_parameters.to_dict()

    if stage_one_compile_parameters is None:
        stage_one_compile_parameters = compile_parameters
    if stage_two_compile_parameters is None:
        stage_two_compile_parameters = compile_parameters


    if isinstance(epochs, tuple):
        first_train_epochs, second_train_epochs = epochs
    else:
        first_train_epochs = epochs
        second_train_epochs = ceil(epochs / 2)


    model.compile(**stage_one_compile_parameters)

    model.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=first_train_epochs,
        validation_data=validation_data,
        shuffle=shuffle
    )

    representation_layer_index = positive_model_layer_index(model, representation_layer_index)

    found_layer, found_index = util.get_representation_layer_index(
        model,
        desired_layer_index=representation_layer_index
    )
    if found_index is None:
        raise ValueError("Unable to find viable representation layer. Please ensure you model has at least two trainable layers")
    if representation_layer_index > found_index:
        warnings.warn(
            f"Overriding representation layer to layer {found_index} (originally {representation_layer_index})")
        representation_layer_index = found_index



    untrainable_layers = model.layers[:representation_layer_index+1]
    trainable_layers = model.layers[representation_layer_index+1:]
    for layer in untrainable_layers:
        layer.trainable = False
    for layer in trainable_layers:
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            layer.set_weights([layer.kernel_initializer(shape=np.asarray(layer.kernel.shape)),
                               layer.bias_initializer(shape=np.asarray(layer.bias.shape))])

    balanced_fit(
        model,
        x,
        y,
        compile_parameters=stage_two_compile_parameters,
        batch_size=batch_size,
        epochs=second_train_epochs,
        validation_data=validation_data,
        shuffle=shuffle,
        mode=mode,
        stratify_batches=stratify_batches,
        sample_weights=sample_weights,
        sample_densities=sample_densities,
        class_weights=class_weights,
    )


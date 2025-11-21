import imbal.classification as classification
import imbal.regression as regression
import imbal.util as util
import numpy as np
import warnings
from math import ceil

def decoupled_fit(
    model,
    x=None,
    y=None,
    compile_parameters=None,
    stage_one_compile_parameters=None,
    stage_two_compile_parameters=None,
    batch_size=32,
    epochs=1,
    validation_data=None,
    shuffle=True,
    representation_layer_index=-2,
    aed_for_representation=True,
    mode='classification'
):
    model.trainable = True

    if isinstance(compile_parameters, util.TFModelCompileParameters):
        compile_parameters = compile_parameters.to_dict()
    if isinstance(stage_one_compile_parameters, util.TFModelCompileParameters):
        stage_one_compile_parameters = stage_one_compile_parameters.to_dict()
    if isinstance(stage_two_compile_parameters, util.TFModelCompileParameters):
        stage_two_compile_parameters = stage_two_compile_parameters.to_dict()

    if stage_one_compile_parameters is None:
        if compile_parameters is None:
            stage_one_compile_parameters = {}
        else:
            stage_one_compile_parameters = compile_parameters
    if stage_two_compile_parameters is None:
        if compile_parameters is None:
            if stage_one_compile_parameters is None:
                stage_two_compile_parameters = {}
            else:
                stage_two_compile_parameters = stage_one_compile_parameters
        else:
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

    if representation_layer_index < 0:
        representation_layer_index =  len(model.layers) + representation_layer_index

    found_layer, found_index = util.get_last_trainable_index(
        model,
        n_to_last=2,
        desired_layer_index=representation_layer_index
    )
    if found_index is None:
        raise ValueError("Unable to find viable representation layer. Please ensure you model has at least two trainable layers")
    if representation_layer_index != found_index:
        representation_layer_index = found_index
        warnings.warn(f"Specified representation layer index is an untrainable layer, "
                      f"overriding to layer {found_index}")

    untrainable_layers = model.layers[:representation_layer_index+1]
    trainable_layers = model.layers[representation_layer_index+1:]
    for layer in untrainable_layers:
        layer.trainable = False
    for layer in trainable_layers:
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            layer.set_weights([layer.kernel_initializer(shape=np.asarray(layer.kernel.shape)),
                               layer.bias_initializer(shape=np.asarray(layer.bias.shape))])

    model.compile(**stage_two_compile_parameters)

    if mode=='classification':
        weights = classification.generate_weights(y)
        dataset = classification.DatasetWithBatching(
            x,
            y,
            sample_weights=weights,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    else:
        weights = regression.generate_weights(y)
        dataset = regression.DatasetWithBatching(
            x,
            y,
            sample_weights=weights,
            batch_size=batch_size,
            shuffle=shuffle,
        )


    print(weights)

    model.fit(
        dataset,
        epochs=second_train_epochs,
        validation_data=validation_data,
    )
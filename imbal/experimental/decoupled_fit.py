import imbal.util as util
import numpy as np
import warnings
from math import ceil
from .balanced_fit import balanced_fit
from imbal.util.backend.tools import positive_model_layer_index, safe_object_unwrap
from imbal.util.backend.fit.generate_decoder_branch import generate_decoder_branch as generate_branch
from imbal.util.backend.fit.generate_decoder_branch import mse_reconstruction_loss

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
    representation_layer_index=-3,
    stratify_batches=True,
    generate_decoder_branch=False,
    multi_input=False,
    multi_output=False,
    output_label_index=0,
    mode='classification'
):

    compiling_model = model
    compiling_model.trainable = True

    compile_parameters = safe_object_unwrap(compile_parameters, util.ModelCompileParameters)

    if isinstance(stage_one_compile_parameters, util.ModelCompileParameters):
        stage_one_compile_parameters = stage_one_compile_parameters.to_dict()
    if isinstance(stage_two_compile_parameters, util.ModelCompileParameters):
        stage_two_compile_parameters = stage_two_compile_parameters.to_dict()

    if stage_one_compile_parameters is None:
        stage_one_compile_parameters = compile_parameters.copy()
    if stage_two_compile_parameters is None:
        stage_two_compile_parameters = compile_parameters.copy()

    if isinstance(epochs, tuple):
        first_train_epochs, second_train_epochs = epochs
    else:
        first_train_epochs = epochs
        second_train_epochs = ceil(epochs / 2)

    generated_decoder_layers = []
    if generate_decoder_branch:
        compiling_model, generated_decoder_layers = generate_branch(compiling_model, representation_layer_index)
        model_loss = compile_parameters.get('loss', None)
        if model_loss is None:
            stage_one_compile_parameters['loss'] = mse_reconstruction_loss
            stage_two_compile_parameters['loss'] = mse_reconstruction_loss
        else:
            stage_one_compile_parameters['loss'] = [compile_parameters['loss'], mse_reconstruction_loss]
            stage_two_compile_parameters['loss'] = [compile_parameters['loss'], mse_reconstruction_loss]

        model_metrics = compile_parameters.get('metrics', None)
        if model_metrics is None:
            stage_one_compile_parameters['metrics'] = ['mse']
            stage_two_compile_parameters['metrics'] = ['mse']
        else:
            if isinstance(model_metrics[0], list) or isinstance(model_metrics[0], tuple):
                stage_one_compile_parameters['metrics'] = compile_parameters['metrics'] + [['mse']]
                stage_two_compile_parameters['metrics'] = compile_parameters['metrics'] + [['mse']]
            else:
                stage_one_compile_parameters['metrics'] = [compile_parameters['metrics']] + [['mse']]
                stage_two_compile_parameters['metrics'] = [compile_parameters['metrics']] + [['mse']]

        if multi_output:
            y.append(x)
        else:
            y = [y, x]

        multi_output = True
        generate_decoder_branch = False


    compiling_model.compile(**stage_one_compile_parameters)

    compiling_model.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=first_train_epochs,
        validation_data=validation_data,
        shuffle=shuffle
    )

    representation_layer_index = positive_model_layer_index(compiling_model, representation_layer_index)

    found_layer, found_index = util.get_representation_layer_index(
        compiling_model,
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
    for layer in generated_decoder_layers:
        layer.trainable = False
    for layer in trainable_layers:
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            layer.set_weights([layer.kernel_initializer(shape=np.asarray(layer.kernel.shape)),
                               layer.bias_initializer(shape=np.asarray(layer.bias.shape))])

    balanced_fit(
        compiling_model,
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
        multi_input=multi_input,
        multi_output=multi_output,
        output_label_index=output_label_index,
        generate_decoder_branch=generate_decoder_branch,
    )

    model.compile(**compile_parameters)


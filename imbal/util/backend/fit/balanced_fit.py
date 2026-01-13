import imbal
from imbal import classification, regression, util
from imbal.util.backend.tools import safe_object_unwrap
import warnings
from imbal.util.backend.fit.generate_decoder_branch import generate_decoder_branch as generate_branch
from imbal.util.backend.fit.generate_decoder_branch import mse_reconstruction_loss

def balanced_fit(
    model,
    x=None,
    y=None,
    compile_parameters=None,
    sample_densities=None,
    class_weights=None,
    sample_weights=None,
    batch_size=32,
    epochs=1,
    validation_data=None,
    shuffle=True,
    mode='classification',
    generate_decoder_branch=False,
    representation_layer_index=-3,
    stratify_batches=True
):

    compiling_model = model
    compile_parameters = safe_object_unwrap(compile_parameters, util.ModelCompileParameters)
    extended_parameters = compile_parameters.copy()

    if generate_decoder_branch:
        compiling_model, _ = generate_branch(compiling_model, representation_layer_index)
        model_loss = compile_parameters.get('loss', None)
        if model_loss is None:
            extended_parameters['loss'] = mse_reconstruction_loss
        else:
            extended_parameters['loss'] = [compile_parameters['loss'], mse_reconstruction_loss]

        model_metrics = compile_parameters.get('metrics', None)
        if model_metrics is None:
            extended_parameters['metrics'] = ['mse']
        else:
            if isinstance(model_metrics[0], list) or isinstance(model_metrics[0], tuple):
                extended_parameters['metrics'] = compile_parameters['metrics'] + [['mse']]
            else:
                extended_parameters['metrics'] = [compile_parameters['metrics']] + [['mse']]

        y = [y, x]


    has_branch = isinstance(y, list) and len(y) == 2

    dataset = x
    if mode == 'classification':
        if sample_weights is not None and class_weights is not None:
            warnings.warn('Both sample_weights and class_weights have been provided' +
                          'to balanced_fit. class_weights will be ignored.')
        if sample_weights is None:
            if has_branch:
                sample_weights = classification.generate_sample_weights(y[0], class_weights=class_weights)
            else:
                sample_weights = classification.generate_sample_weights(y, class_weights=class_weights)
        if stratify_batches:
            if has_branch:
                dataset = imbal.util.backend.MultiDatasetWithBatching(
                    x,
                    y,
                    sample_weights=sample_weights,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    multi_output=True,
                    output_label_index=0,
                    mode='classification'
                )
            else:
                dataset = classification.DatasetWithBatching(
                    x,
                    y,
                    sample_weights=sample_weights,
                    batch_size=batch_size,
                    shuffle=shuffle
                )
    else:
        if sample_weights is not None and sample_densities is not None:
            warnings.warn('Both sample_weights and sample_densities have been provided' +
                          'to balanced_fit. sample_densities will be ignored.')
        if sample_weights is None:
            if sample_densities is None:
                raise ValueError('Must provide either sample_densities or sample_weights')
            sample_weights = regression.generate_sample_weights(sample_densities)

        if stratify_batches:
            if has_branch:
                dataset = imbal.util.backend.MultiDatasetWithBatching(
                    x,
                    y,
                    sample_weights=sample_weights,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    multi_output=True,
                    output_label_index=0,
                    mode='regression'
                )
            else:
                dataset = regression.DatasetWithBatching(
                    x,
                    y,
                    sample_weights=sample_weights,
                    batch_size=batch_size,
                    shuffle=shuffle
                )


    compiling_model.compile(**extended_parameters)

    compiling_model.fit(
        x=dataset,
        y=None if stratify_batches else y,
        sample_weight=None if stratify_batches else sample_weights,
        epochs=epochs,
        validation_data=validation_data
    )

    model.compile(**compile_parameters)
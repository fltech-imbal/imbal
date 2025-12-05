from imbal import classification, regression, util
from imbal.util.backend.tools import safe_object_unwrap
import warnings

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
    stratify_batches=True
):

    compile_parameters = safe_object_unwrap(compile_parameters, util.ModelCompileParameters)

    model.compile(**compile_parameters)

    dataset = x
    if mode == 'classification':
        if sample_weights is not None and class_weights is not None:
            warnings.warn('Both sample_weights and class_weights have been provided' +
                          'to balanced_fit. class_weights will be ignored.')
        if sample_weights is None:
            sample_weights = classification.generate_sample_weights(y, class_weights=class_weights)
        if stratify_batches:
            dataset = classification.DatasetWithBatching(
                x,
                y,
                sample_weights=sample_weights,
                batch_size=batch_size,
                shuffle=shuffle,
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
            dataset = regression.DatasetWithBatching(
                x,
                y,
                sample_weights=sample_weights,
                batch_size=batch_size,
                shuffle=shuffle,
            )

    model.fit(
        x=dataset if stratify_batches else x,
        y=None if stratify_batches else y,
        sample_weight=None if stratify_batches else sample_weights,
        epochs=epochs,
        validation_data=validation_data
    )
from imbal import classification, regression, util
from imbal.util.backend.tools import safe_object_unwrap

def balanced_fit(
    model,
    x=None,
    y=None,
    compile_parameters=None,
    sample_densities=None,
    sample_weights=None,
    class_weights=None,
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
        if sample_weights is None:
            sample_weights = classification.generate_weights(y, class_weights=class_weights)
        if stratify_batches:
            dataset = classification.DatasetWithBatching(
                x,
                y,
                sample_weights=sample_weights,
                batch_size=batch_size,
                shuffle=shuffle,
            )
    else:
        if sample_weights is None:
            if sample_densities is None:
                raise ValueError('Must provide either sample_densities or sample_weights')
            sample_weights = regression.generate_weights(sample_densities)

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
import imbal
import numpy as np

def decoupled_fit(
    model,
    x=None,
    y=None,
    compile_function=None,
    batch_size=32,
    epochs=1,
    validation_data=None,
    shuffle=True,
    representation_layer_index=-2,
    aed_for_representation=True
):
    model.trainable = True

    compile_function(model, 1)

    model.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        shuffle=shuffle
    )

    untrainable_layers = model.layers[:representation_layer_index+1]
    trainable_layers = model.layers[representation_layer_index+1:]
    for layer in untrainable_layers:
        layer.trainable = False
    for layer in trainable_layers:
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            layer.set_weights([layer.kernel_initializer(shape=np.asarray(layer.kernel.shape)),
                               layer.bias_initializer(shape=np.asarray(layer.bias.shape))])

    compile_function(model, 2)

    weights = imbal.classification.generate_weights(y)

    dataset = imbal.classification.DatasetWithBatching(
        x,
        y,
        sample_weights=weights,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    model.fit(
        dataset,
        epochs=epochs,
        validation_data=validation_data,
    )
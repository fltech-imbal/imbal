import imbal

def decoupled_fit(
    model,
    x=None,
    y=None,
    compile_parameters=None,
    batch_size=32,
    epochs=1,
    validation_data=None,
    shuffle=True,
    representation_layer_index=-2,
    aed_for_representation=True
):
    imbal.util.decoupled_fit(
        model,
        x=x,
        y=y,
        compile_parameters=compile_parameters,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        shuffle=shuffle,
        representation_layer_index=representation_layer_index,
        aed_for_representation=aed_for_representation,
        mode='regression'
    )
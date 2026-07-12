from tensorflow import keras
from tensorflow.keras import layers
import imbal

def build_sep_ec_model(
        input_shape,
        layer_dims,
        unit=False,
        extra_regressor_layers=False
):
    inputs = keras.Input(shape=(input_shape[1],))
    x = inputs
    representation_layer = None
    pre_representation_layer = None
    for index, num_units in enumerate(layer_dims):
        if index == len(layer_dims) - 2:
            x = layers.Dense(num_units, activation='relu', name='pre_representation' if not unit else None)(x)
            pre_representation_layer = x
        elif index == len(layer_dims) - 1:
            x = layers.Dense(num_units, name='representation' if not unit else None)(x)
            if unit:
                x = layers.UnitNormalization(name='representation')(x)
            representation_layer = x
        else:
            x = layers.Dense(num_units, activation='relu')(x)

    if extra_regressor_layers:
        x = layers.Dense(64, activation='relu')(x)
        output_layer = layers.Dense(1)(x)
    else:
        output_layer = layers.Dense(1)(x)

    return keras.Model(inputs=inputs, outputs=[output_layer, representation_layer, pre_representation_layer], name="SEP_EC")
from tensorflow import keras
from tensorflow.keras import layers
import imbal

def build_sep_ec_model(input_dim):
    HIDDEN_LAYERS = [2048, 128, 1024, 128, 512, 128, 256, 128]
    EMBED_DIM = 128
    DROPOUT_RATE = 0.20

    inputs = keras.Input(shape=(input_dim,))

    x = inputs
    for num_units in HIDDEN_LAYERS:
        x = layers.Dense(num_units, activation='relu')(x)
        x = layers.Dropout(DROPOUT_RATE)(x)

    # x = layers.Flatten()(x)
    x = layers.Dense(EMBED_DIM, activation='relu', name="embedding")(x)
#     x = layers.Flatten()(x)
    outputs = layers.Dense(1)(x)

    model = imbal.regression.Model(inputs=inputs, outputs=outputs, name="SEP_EC")

    return model
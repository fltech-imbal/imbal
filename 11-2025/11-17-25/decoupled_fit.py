import keras
from keras import layers
import numpy as np
from keras.utils import to_categorical

MODE = 'classification'
IMBALANCED = False
HIGH_IMBALANCE = False
SAVE_FIG_NAME = f'tsne-{MODE}-imbalanced-{IMBALANCED}.png'

num_classes = 10
input_shape = (28, 28, 1)

batch_size = 128
epochs = 10

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

inputs = keras.Input(shape=input_shape)
x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

from imbal import classification

def compile_model(model, stage):
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

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
            print(layer)
            layer.set_weights([layer.kernel_initializer(shape=np.asarray(layer.kernel.shape)),
                               layer.bias_initializer(shape=np.asarray(layer.bias.shape))])

    compile_function(model, 2)

    weights = classification.generate_weights(y)

    dataset = classification.DatasetWithBatching(
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

decoupled_fit(model, x_train, y_train, compile_function=compile_model, epochs=10, batch_size=512)

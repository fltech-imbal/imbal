import keras
from keras import layers
import numpy as np
from keras.utils import to_categorical
from scipy.stats.tests.test_continuous_fit_censored import optimizer

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

import imbal

parameters = imbal.classification.compile_parameters(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"]
)

# parameters = {
#     'loss' : 'categorical_crossentropy',
#     'optimizer': 'adam',
#     'metrics' : ['accuracy'],
# }

imbal.classification.decoupled_fit(
    model,
    x_train,
    y_train,
    compile_parameters=parameters,
    epochs=(10, 5),
    batch_size=512
)

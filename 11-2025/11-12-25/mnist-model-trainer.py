import keras
from keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np

MODE = 'classification'
IMBALANCED = False
HIGH_IMBALANCE = False
SAVE_FIG_NAME = f'tsne-{MODE}-imbalanced-{IMBALANCED}.png'


"""
Load STL-10 Dataset
"""


(x_train, y_train), (x_test, y_test) = mnist.load_data()

inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(16, (3, 3), activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(
    10,
    activation='softmax'
)(x)

model = keras.Model(inputs=inputs, outputs=output)

loss_fn = ('sparse_categorical_crossentropy' if HIGH_IMBALANCE == False else 'binary_crossentropy') if MODE == 'classification' else 'mse'
metrics = ['accuracy'] if MODE == 'classification' else ['mae', 'mse']

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=loss_fn,
              metrics=metrics,)

epochs = 15 if MODE == 'classification' else 15

history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

model.save_weights(f'mnist-trained-{MODE}-model-imbalanced-{IMBALANCED}.weights.h5')
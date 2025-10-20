from tensorflow.keras.datasets import mnist
import keras
from keras import layers
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = np.where(y_train == 0, 0, 1)
y_test = np.where(y_test == 0, 0, 1)


inputs = keras.Input(shape=(28,28))
flatten = layers.Flatten()(inputs)
hidden = layers.Dense(32, activation='relu')(flatten)
output = layers.Dense(1, activation='softmax')(hidden)

model = keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=["accuracy"])

from imbal.classification import DatasetWithBatching, generate_weights

sampler = DatasetWithBatching(
    x_train,
    y_train,
    batch_size=512,
    sample_weights=generate_weights(y_train),
)

history = model.fit(sampler, epochs=30)

import imbal

imbal.classification.tsne_visualization(model, x_test, y_test, save_figure='test.png')



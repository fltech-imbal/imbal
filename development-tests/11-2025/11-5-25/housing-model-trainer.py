import keras
from keras import layers, regularizers
import numpy as np
from random import randint
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

MODE = 'regression'
IMBALANCED = False
HIGH_IMBALANCE = False
SAVE_FIG_NAME = f'tsne-{MODE}-imbalanced-{IMBALANCED}.png'


"""
Load Housing Dataset
"""
x, y = fetch_california_housing(return_X_y=True)

DATASET_PERCENTAGE = 1.0
TRAIN_SPLIT = 0.8

num_data = x.shape[0]
percent_index = int(num_data * DATASET_PERCENTAGE)
shuffled_indices = np.random.permutation(len(x))[:percent_index]
x = x[shuffled_indices]
y = y[shuffled_indices]
num_data = x.shape[0]
split_index = int(num_data * TRAIN_SPLIT)
x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
print('x_train', x_train.shape)
print('y_train',y_train.shape)
print('x_test',x_test.shape)
print('y_test',y_test.shape)


inputs = keras.Input(shape=(x.shape[1],))
output = None
if MODE == 'classification':
    x = layers.Conv2D(16, (3, 3), activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(
        10 if HIGH_IMBALANCE == False else 1,
        activation='softmax' if HIGH_IMBALANCE == False else 'sigmoid'
    )(x)
else:
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1)(x)

loss_fn = ('sparse_categorical_crossentropy' if HIGH_IMBALANCE == False else 'binary_crossentropy') if MODE == 'classification' else 'mse'
metrics = ['accuracy'] if MODE == 'classification' else ['mae', 'mse']


model = keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=metrics)

from imbal.classification import DatasetWithBatching

sampler = DatasetWithBatching(
    x_train,
    y_train,
    batch_size=256,
)

epochs = 50 if MODE == 'classification' else 15

history = model.fit(sampler, epochs=epochs, validation_data=(x_test, y_test))

model.save_weights(f'housing-trained-tabular-model-{MODE}.weights.h5')
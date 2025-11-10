import keras
from keras import layers, regularizers
import numpy as np
from sklearn.datasets import load_wine

MODE = 'classification'
IMBALANCED = False
HIGH_IMBALANCE = False
SAVE_FIG_NAME = f'tsne-{MODE}-imbalanced-{IMBALANCED}.png'


"""
Load Housing Dataset
"""
x, y = load_wine(return_X_y=True)
shuffle = np.random.permutation(len(x))
x = x[shuffle]
y = y[shuffle]

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
x = layers.Dense(32, activation='relu')(inputs)
x = layers.Dense(16, activation='relu')(x)
output = layers.Dense(3, activation='softmax')(x)

loss_fn = 'sparse_categorical_crossentropy'
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

epochs = 250 if MODE == 'classification' else 15

history = model.fit(sampler, epochs=epochs, validation_data=(x_test, y_test))

model.save_weights(f'wine-trained-tabular-model-{MODE}.weights.h5')
import keras
from keras import layers, regularizers
import numpy as np
from random import randint

MODE = 'regression'
IMBALANCED = False
HIGH_IMBALANCE = False
SAVE_FIG_NAME = f'tsne-{MODE}-imbalanced-{IMBALANCED}.png'


"""
Load STL-10 Dataset
"""


DATASET_PERCENTAGE = 0.8
TRAIN_SPLIT = 0.8

with open('../../../stl10_binary/train_X.bin', 'rb') as file:
    everything = np.fromfile(file, dtype=np.uint8)
    x_train = np.reshape(everything, (-1, 3, 96, 96))
    x_train = np.transpose(x_train, (0, 3, 2, 1))
with open('../../../stl10_binary/train_y.bin', 'rb') as file:
    y_train = np.fromfile(file, dtype=np.uint8)
with open('../../../stl10_binary/test_X.bin', 'rb') as file:
    everything = np.fromfile(file, dtype=np.uint8)
    x_test = np.reshape(everything, (-1, 3, 96, 96))
    x_test = np.transpose(x_test, (0, 3, 2, 1))
with open('../../../stl10_binary/test_y.bin', 'rb') as file:
    y_test = np.fromfile(file, dtype=np.uint8)
x_combined = np.concatenate((x_train, x_test), axis=0) / 255.0
y_combined = np.concatenate((y_train, y_test), axis=0) - 1

num_data = x_combined.shape[0]
percent_index = int(num_data * DATASET_PERCENTAGE)
shuffled_indices = np.random.permutation(len(x_combined))[:percent_index]
x_combined = x_combined[shuffled_indices]
y_combined = y_combined[shuffled_indices]
num_data = x_combined.shape[0]
split_index = int(num_data * TRAIN_SPLIT)
x_train, x_test = x_combined[:split_index], x_combined[split_index:]
y_train, y_test = y_combined[:split_index], y_combined[split_index:]
print('x_train', x_train.shape)
print('y_train',y_train.shape)
print('x_test',x_test.shape)
print('y_test',y_test.shape)


if IMBALANCED:
    if HIGH_IMBALANCE:
        if MODE == 'classification':
            train_by_class = [x_train[y_train == i] for i in range(10)]
            x_train = np.concatenate(train_by_class, axis=0)
            y_train = np.concatenate([
                np.full(len(train_by_class[i]), 0 if i == 0 else 1) for i in range(10)
            ])
            test_by_class = [x_test[y_test == i] for i in range(10)]
            x_test = np.concatenate(test_by_class, axis=0)
            y_test = np.concatenate([
                np.full(len(test_by_class[i]), 0 if i == 0 else 1) for i in range(10)
            ])
        else:
            train_by_class = [x_train[y_train == i] for i in range(10)][:5*randint(90,105)]
            x_train = np.concatenate(train_by_class, axis=0)
            y_train = np.concatenate([
                np.full(len(train_by_class[i]), i if i < 4 else 4) for i in range(10)
            ])
            test_by_class = [x_test[y_test == i] for i in range(10)][:randint(90,105)]
            x_test = np.concatenate(test_by_class, axis=0)
            y_test = np.concatenate([
                np.full(len(test_by_class[i]), i if i < 4 else 4) for i in range(10)
            ])
    else:
        train_by_class = [x_train[y_train == i] for i in range(10)]
        for i in range(10):
            train_by_class[i] = train_by_class[i][:500 * (i + 1)]
        x_train = np.concatenate(train_by_class, axis=0)
        y_train = np.concatenate([
            np.full(len(train_by_class[i]), i) for i in range(10)
        ])

        test_by_class = [x_test[y_test == i] for i in range(10)]
        for i in range(10):
            test_by_class[i] = test_by_class[i][:90*(i+1)]
        x_test = np.concatenate(test_by_class, axis=0)
        y_test = np.concatenate([
            np.full(len(test_by_class[i]), i) for i in range(10)
        ])

inputs = keras.Input(shape=(96, 96, 3))
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

epochs = 50 if MODE == 'classification' else 30

history = model.fit(sampler, epochs=epochs, validation_data=(x_test, y_test))

model.save_weights(f'stl-10-trained-{MODE}-model-imbalanced-{IMBALANCED}.weights.h5')
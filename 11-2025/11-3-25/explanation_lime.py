from tensorflow.keras.datasets import mnist
import keras
import numpy as np
from random import randint
import os
import csv

import imbal.util.explanation

LIME_MODE = 'image'
MODE = 'classification'

IMBALANCED = False
HIGH_IMBALANCE = False
SAVE_FIG_NAME = f'tsne-{MODE}-imbalanced-{IMBALANCED}.png'

if LIME_MODE == 'image':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
else:
    def read_csv_to_list_of_lists(filepath):
        data = []
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                data.append(row)
        return data


    PATH_START = '/mnt/c/Users/tommy/Desktop/Repos/dr-chan-work-demo'
    print(os.getcwd())

    # data = np.array(read_csv_to_list_of_lists(f'{PATH_START}/CISIR-data/SARCOS/sarcos_inv_training.csv'))
    # print(data.shape)
    # data = data[1:, -1].astype(float)

    # data = np.array(read_csv_to_list_of_lists(f'{PATH_START}/CISIR-data/SEP-C/sep_10mev_training.csv'))
    # print(data.shape)
    # data = data[1:, 22].astype(float)

    data = np.array(
        read_csv_to_list_of_lists(f'{PATH_START}/CISIR-data/SEP-EC/training/sep_event_1_filled_ie_trim.csv'))[1:]
    for i in range(43):
        if os.path.exists(f'{PATH_START}/CISIR-data/SEP-EC/training/sep_event_{i + 2}_filled_ie_trim.csv'):
            data = np.concatenate([data, read_csv_to_list_of_lists(
                f'{PATH_START}/CISIR-data/SEP-EC/training/sep_event_{i + 2}_filled_ie_trim.csv')[1:]])
    print(data.shape)
    data = np.concatenate([data[:, 3:161].astype(float), data[:, 162:].astype(float)], axis=1)
    labels = data[:, -1].astype(float)

    print(labels.min(), labels.max())

    if MODE == 'classification':
        labels = np.array([[0] if x < 1 else [1] for x in labels])

    print(data.shape)

    length = data.shape[0]
    train_split = 0.8

    x_train, y_train = data[:int(length * train_split)], labels[:int(length * train_split)]
    x_test, y_test = data[int(length * train_split):], labels[int(length * train_split):]

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

from keras import layers

if LIME_MODE == 'image':
    inputs = keras.Input(shape=(28,28,1))
    output = None
    if MODE == 'classification':
        flatten = layers.Flatten()(inputs)
        hidden = layers.Dense(32, activation='relu')(flatten)
        output = layers.Dense(
            10 if HIGH_IMBALANCE == False else 1,
            activation='softmax' if HIGH_IMBALANCE == False else 'sigmoid'
        )(hidden)
    else:
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(1)(x)

    loss_fn = ('sparse_categorical_crossentropy' if HIGH_IMBALANCE == False else 'binary_crossentropy') if MODE == 'classification' else 'mse'
    metrics = ['accuracy'] if MODE == 'classification' else ['mae', 'mse']
    optimizer = 'adam'
else:
    inputs = keras.Input(shape=(179, 1))
    flatten = layers.Flatten()(inputs)
    hidden = layers.Dense(64, activation='relu')(flatten)
    hidden_2 = layers.Dense(32, activation='relu')(hidden)
    output = layers.Dense(1)(hidden_2)

    loss_fn = 'mse'
    metrics = ['mae', 'mse']
    optimizer = keras.optimizers.Adam(learning_rate=2e-4)

model = keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=metrics)

if LIME_MODE == 'image':
    model.load_weights(f'trained-{MODE}-model-imbalanced-{IMBALANCED}.weights.h5')
else:
    model.load_weights(f'trained-tabular-model-{MODE}.weights.h5')

EXPLAIN_INDEX = 1300

if LIME_MODE=='image':
    x_test = np.stack((x_test,)*3, axis=-1)

print(x_test.shape)

imbal.util.explanation.lime_explanation(
    x_test,
    y_test,
    model,
    instance_index=EXPLAIN_INDEX,
    lime_mode=LIME_MODE,
    model_type=MODE,
    num_features=2
)


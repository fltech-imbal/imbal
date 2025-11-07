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
    DATASET_PERCENTAGE = 0.8
    TRAIN_SPLIT = 0.8

    with open('../../stl10_binary/train_X.bin', 'rb') as file:
        everything = np.fromfile(file, dtype=np.uint8)
        x_train = np.reshape(everything, (-1, 3, 96, 96))
        x_train = np.transpose(x_train, (0, 3, 2, 1))
    with open('../../stl10_binary/train_y.bin', 'rb') as file:
        y_train = np.fromfile(file, dtype=np.uint8)
    with open('../../stl10_binary/test_X.bin', 'rb') as file:
        everything = np.fromfile(file, dtype=np.uint8)
        x_test = np.reshape(everything, (-1, 3, 96, 96))
        x_test = np.transpose(x_test, (0, 3, 2, 1))
    with open('../../stl10_binary/test_y.bin', 'rb') as file:
        y_test = np.fromfile(file, dtype=np.uint8)
    x_combined = np.concatenate((x_train, x_test), axis=0) / 255.0
    y_combined = np.concatenate((y_train, y_test), axis=0) - 1

    num_data = x_combined.shape[0]
    percent_index = int(num_data * DATASET_PERCENTAGE)
    x_combined = x_combined[:percent_index]
    y_combined = y_combined[:percent_index]
    num_data = x_combined.shape[0]
    split_index = int(num_data * TRAIN_SPLIT)
    x_train, x_test = x_combined[:split_index], x_combined[split_index:]
    y_train, y_test = y_combined[:split_index], y_combined[split_index:]
    print('x_train', x_train.shape)
    print('y_train', y_train.shape)
    print('x_test', x_test.shape)
    print('y_test', y_test.shape)
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
    inputs = keras.Input(shape=(96,96,3))
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

model.load_weights(f'stl-10-trained-{MODE}-model-imbalanced-{IMBALANCED}.weights.h5')

print(x_test.shape)
result = model.predict(np.reshape(x_test[0], (1, 96, 96, 3)))
print('test', result[0])

class_labels = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']



EXPLAIN_INDEX_START = 20
EXPLAIN_AMOUNT = 50

for i in range(6):
    x_ = x_test[EXPLAIN_INDEX_START]
    y_ = y_test[EXPLAIN_INDEX_START]

    imbal.classification.explanation.lime_image_explanation(
        x_,
        model,
        label=y_,
        num_features=10**i,
        class_names=class_labels
    )

for i in range(EXPLAIN_AMOUNT):
    x_ = x_test[i + EXPLAIN_INDEX_START]
    y_ = y_test[i + EXPLAIN_INDEX_START]

    imbal.classification.explanation.lime_image_explanation(
        x_,
        model,
        label=y_,
        class_names=class_labels
    )


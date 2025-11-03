import os
import numpy as np
import csv
from sklearn.neighbors import KernelDensity
import imbal

MODE='classification'

def read_csv_to_list_of_lists(filepath):
    data = []
    with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            data.append(row)
    return data

PATH_START = '/mnt/c/Users/tommy/PycharmProjects/DrChanWorkPlayground'
print(os.getcwd())

# data = np.array(read_csv_to_list_of_lists(f'{PATH_START}/CISIR-data/SARCOS/sarcos_inv_training.csv'))
# print(data.shape)
# data = data[1:, -1].astype(float)

# data = np.array(read_csv_to_list_of_lists(f'{PATH_START}/CISIR-data/SEP-C/sep_10mev_training.csv'))
# print(data.shape)
# data = data[1:, 22].astype(float)

data = np.array(read_csv_to_list_of_lists(f'{PATH_START}/CISIR-data/SEP-EC/training/sep_event_1_filled_ie_trim.csv'))[1:]
for i in range(43):
    if os.path.exists(f'{PATH_START}/CISIR-data/SEP-EC/training/sep_event_{i+2}_filled_ie_trim.csv'):
        data = np.concatenate([data, read_csv_to_list_of_lists(f'{PATH_START}/CISIR-data/SEP-EC/training/sep_event_{i+2}_filled_ie_trim.csv')[1:]])
print(data.shape)
data = np.concatenate([data[:, 3:161].astype(float), data[:, 162:].astype(float)], axis=1)
labels = data[:, -1].astype(float)

print(labels.min(), labels.max())

if MODE == 'classification':
    labels = np.array([[0] if x < 1 else [1] for x in labels])

print(data.shape)

length = data.shape[0]
train_split = 0.8

x_train, y_train = data[:int(length*train_split)], labels[:int(length*train_split)]
x_test, y_test = data[int(length*train_split):], labels[int(length*train_split):]

import keras
from keras import layers

inputs = keras.Input(shape=(179,1))
flatten = layers.Flatten()(inputs)
hidden = layers.Dense(64, activation='relu')(flatten)
hidden_2 = layers.Dense(32, activation='relu')(hidden)
output = layers.Dense(1, activation=None if MODE=='regression' else 'sigmoid')(hidden_2)

loss_fn = 'mse' if MODE=='regression' else 'binary_crossentropy'
metrics = ['accuracy'] if MODE=='classification' else ['mae', 'mse']
optimizer = keras.optimizers.Adam(learning_rate=2e-4 if MODE=='regression' else 5e-5)

model = keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=metrics)

if MODE=='classification':
    from imbal.classification import DatasetWithBatching, generate_weights
    weights = generate_weights(y_train)
    print(np.sum(weights))

    print(x_train.shape)
    print(y_train.shape)
    print(weights.shape)

    sampler = DatasetWithBatching(
        x_train,
        y_train,
        sample_weights=weights,
        batch_size=512,
    )
else:
    from imbal.regression import DatasetWithBatching, generate_weights, get_densities, fit_kde

    kde_bandwidth = fit_kde(
        y_train,
        bin_count=64,
        tolerance=1e-4
    )

    densities = get_densities(
        y_train,
        kde_bandwidth,
        atol=1e-4
    )
    weights = generate_weights(densities)
    print(np.sum(weights))

    sampler = DatasetWithBatching(
        x_train,
        y_train,
        sample_weights=weights,
        batch_size=512,
    )

epochs = 150

history = model.fit(sampler, epochs=epochs)

model.save_weights(f'trained-tabular-model-{MODE}.weights.h5')
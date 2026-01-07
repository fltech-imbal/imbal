import keras
from sklearn.datasets import load_wine
import numpy as np

import imbal.backend.explanation
MODE='classification'
IMBALANCED = False
HIGH_IMBALANCE = False
SAVE_FIG_NAME = f'tsne-{MODE}-imbalanced-{IMBALANCED}.png'

x, y = load_wine(return_X_y=True)
shuffle = np.random.permutation(len(x))
x = x[shuffle]
y = y[shuffle]

labels = load_wine().feature_names
print(labels)

DATASET_PERCENTAGE = 1.0
TRAIN_SPLIT = 0.8

num_data = x.shape[0]
percent_index = int(num_data * DATASET_PERCENTAGE)
x = x[:percent_index]
y = y[:percent_index]
num_data = x.shape[0]
split_index = int(num_data * TRAIN_SPLIT)
x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
print('x_train', x_train.shape)
print('y_train',y_train.shape)
print('x_test',x_test.shape)
print('y_test',y_test.shape)


from keras import layers

inputs = keras.Input(shape=(x.shape[1],))
x = layers.Dense(32, activation='relu')(inputs)
x = layers.Dense(16, activation='relu')(x)
output = layers.Dense(3, activation='softmax')(x)

loss_fn = 'mse'
metrics = ['mae', 'mse']
optimizer = keras.optimizers.Adam(learning_rate=2e-4)
model = keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=metrics)

model.load_weights(f'wine-trained-tabular-model-{MODE}.weights.h5')

import matplotlib.pyplot as plt

print(x_test.shape)
import numpy as np
indices = np.random.permutation(len(x_test))
x_test = x_test[indices]
y_test = y_test[indices]

print(y_test)

EXPLAIN_INDEX_START = 0
EXPLAIN_AMOUNT = 5

for i in range(EXPLAIN_AMOUNT):
    x = x_test[i + EXPLAIN_INDEX_START]
    y = y_test[i + EXPLAIN_INDEX_START]

    imbal.classification.lime_explain_tabular_sample(
        x,
        model,
        x_train,
        # label_to_explain=y,
        actual_label=y,
        class_names=['Region 1', 'Region 2', 'Region 3'],
        feature_names=labels,
        figure_save_path=f'temp-{i}.html',
        # use_pyplot=True,
        # return_figure=True
    )

    imbal.classification.lime_explain_tabular_sample(
        x,
        model,
        x_train,
        label_to_explain=y,
        actual_label=y,
        class_names=['Region 1', 'Region 2', 'Region 3'],
        feature_names=labels,
        figure_save_path=f'temp-{i}-correct.html',
        # use_pyplot=True,
        # return_figure=True
    )


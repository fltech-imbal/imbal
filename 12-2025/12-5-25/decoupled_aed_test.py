import keras
from tensorflow.keras import layers
import numpy as np
from keras.utils import to_categorical
import time
import tensorflow as tf

MODE = 'decoupled'
FILTER = 'binary'

"""
Load data
"""

num_classes = 10
input_shape = (32, 32, 3)

DATASET_PERCENTAGE = 0.8
TRAIN_SPLIT = 0.8

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_combined = np.concatenate((x_train, x_test), axis=0)
y_combined = np.concatenate((y_train, y_test), axis=0)
y_combined = y_combined.reshape(-1,)

print(x_combined.shape)
print(y_combined.shape)

num_data = x_combined.shape[0]
percent_index = int(num_data * DATASET_PERCENTAGE)
num_data = x_combined.shape[0]
split_index = int(num_data * TRAIN_SPLIT)
x_train, x_test = x_combined[:split_index], x_combined[split_index:]
y_train, y_test = y_combined[:split_index], y_combined[split_index:]
print('x_train', x_train.shape)
print('y_train',y_train.shape)
print('x_test',x_test.shape)
print('y_test',y_test.shape)

class_split = []
for i in range(num_classes):
    class_split.append(len(y_train[y_train == i]))
print(class_split)

x_train_filter = []
y_train_filter = []
x_test_filter = []
y_test_filter = []
print(np.tile(x_train[y_train==0], [1, 1, 1, 1]).shape)
for i in range(num_classes):
    if FILTER == 'binary':
        if i == 6:
            break
        if i == 0:
            x_train_filter.append(np.tile(x_train[y_train == i], [1, 1, 1, 1]))
            y_train_filter.append(np.tile(y_train[y_train == i], 1))
            x_test_filter.append(np.tile(x_test[y_test == i], [1, 1, 1, 1]))
            y_test_filter.append(np.tile(y_test[y_test == i], 1))
        else:
            if i < 5:
                continue
            x_train_filter.append(np.ones(x_train[y_train == i][:40].shape))
            y_train_filter.append(np.ones(y_train[y_train == i][:40].shape))
            x_test_filter.append(np.ones(x_test[y_test == i][:10].shape))
            y_test_filter.append(np.ones(y_test[y_test == i][:10].shape))
    elif i < 1 or FILTER != 'imbalanced':
        x_train_filter.append(np.tile(x_train[y_train==i], [1, 1, 1, 1]))
        y_train_filter.append(np.tile(y_train[y_train==i], 1))
        x_test_filter.append(np.tile(x_test[y_test==i], [1, 1, 1, 1]))
        y_test_filter.append(np.tile(y_test[y_test==i], 1))
    else:
        x_train_filter.append(x_train[y_train == i][:10])
        y_train_filter.append(y_train[y_train == i][:10])
        x_test_filter.append(x_test[y_test == i][:10])
        y_test_filter.append(y_test[y_test == i][:10])

x_train = np.concatenate(x_train_filter)
x_test = np.concatenate(x_test_filter)
y_train = np.concatenate(y_train_filter)
y_test = np.concatenate(y_test_filter)

train_shuffle = np.random.permutation(x_train.shape[0])
test_shuffle = np.random.permutation(x_test.shape[0])
x_train = x_train[train_shuffle]
x_test = x_test[test_shuffle]
y_train = y_train[train_shuffle]
y_test = y_test[test_shuffle]

print(y_train.shape)
print(x_test.shape)

class_split = []
for i in range(num_classes):
    class_split.append(len(y_train[y_train == i]))
print('distribution', class_split)

y_train = to_categorical(y_train, num_classes if FILTER != 'binary' else 2)
y_test = to_categorical(y_test, num_classes if FILTER != 'binary' else 2)


"""
Build model
"""

inputs = keras.Input(shape=input_shape)
x = layers.Conv2D(16, (3, 3), strides=(2, 2))(inputs)
x = layers.LayerNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(32, (3, 3), strides=(2, 2))(x)
x = layers.LayerNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(64, (3, 3), strides=(2, 2))(x)
x = layers.LayerNormalization()(x)
x = layers.Activation('relu')(x)
latent = layers.Flatten()(x)
x = layers.Dropout(0.4)(latent)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(num_classes if FILTER != 'binary' else 2, activation='softmax')(x)

y = layers.Reshape((64, 3, 3))(latent)
y = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(latent)
y = layers.LayerNormalization()(y)
y = layers.Activation('relu')(y)
y = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(y)
y = layers.LayerNormalization()(y)
y = layers.Activation('relu')(y)
y = layers.Conv2D(1, (3, 3))(y)
input_recon = layers.Activation('relu')(y)


model = keras.Model(inputs=inputs, outputs=[output, input_recon])

model.summary()

import imbal

batch_size = 512
epochs = 30

print('number of layers', len(model.layers))

auc = keras.metrics.AUC(multi_label=True)
f1 = tf.keras.metrics.F1Score()

parameters = imbal.classification.wrap_model_compile_parameters(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    metrics=["accuracy", f1, auc]
)

model.compile(**parameters.to_dict())
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs
)



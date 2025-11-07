import keras
from keras import layers, regularizers, models
import numpy as np
from random import randint

MODE = 'classification'
IMBALANCED = False
HIGH_IMBALANCE = False
SAVE_FIG_NAME = f'tsne-{MODE}-imbalanced-{IMBALANCED}.png'


"""
Load STL-10 Dataset
"""


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

# inputs = keras.Input(shape=(96, 96, 3))
# x = layers.Conv2D(16, (3, 3), activation='relu')(inputs)
# x = layers.BatchNormalization()(x)
# x = layers.Conv2D(16, (3, 3), activation='relu')(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv2D(32, (3, 3), activation='relu')(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv2D(32, (3, 3), activation='relu')(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.Conv2D(64, (3, 3), activation='relu')(x)
# x = layers.BatchNormalization()(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.Flatten()(x)
# x = layers.Dense(64, activation='relu')(x)
# x = layers.Dropout(0.3)(x)
# output = layers.Dense(
#     10,
#     activation='softmax'
# )(x)

# model = keras.Model(inputs=inputs, outputs=output)

loss_fn = ('sparse_categorical_crossentropy' if HIGH_IMBALANCE == False else 'binary_crossentropy') if MODE == 'classification' else 'mse'
metrics = ['accuracy'] if MODE == 'classification' else ['mae', 'mse']

from tensorflow.keras.applications import ResNet50

resnet_model = ResNet50(
    include_top=False,      # or False for feature extraction
    weights='imagenet',    # or None for random init
    input_shape=(96, 96, 3)
)

resnet_model.summary()

# Freeze everything first
for layer in resnet_model.layers:
    layer.trainable = False

# Unfreeze the last convolutional block (e.g., 'conv5_block3_out')
set_trainable = False
for layer in resnet_model.layers:
    if layer.name.startswith("conv5_block3"):
        set_trainable = True
    if set_trainable:
        layer.trainable = True

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
])

inputs = keras.Input(shape=(96, 96, 3))
x = data_augmentation(inputs)
x = resnet_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
output = layers.Dense(10, activation='softmax',
             kernel_regularizer=regularizers.l2(1e-4))(x)

model = models.Model(inputs=inputs, outputs=output)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=loss_fn,
              metrics=metrics,)

from imbal.classification import DatasetWithBatching

# sampler = DatasetWithBatching(
#     x_train,
#     y_train,
#     batch_size=256,
# )

epochs = 50 if MODE == 'classification' else 15

history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

model.save_weights(f'stl-10-trained-{MODE}-model-imbalanced-{IMBALANCED}.weights.h5')
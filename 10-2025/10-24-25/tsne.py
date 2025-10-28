from tensorflow.keras.datasets import mnist
import keras
from keras import layers
import numpy as np

MODE = 'classification'
IMBALANCED = True
SAVE_FIG_NAME = f'test-tsne-imbalanced-{MODE}-imbalanced-{IMBALANCED}.png'

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if IMBALANCED:
    train_by_class = [x_train[y_train == i] for i in range(10)]
    for i in range(10):
        train_by_class[i] = train_by_class[i][:500*(i+1)]
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


inputs = keras.Input(shape=(28,28,1))
output = None
if MODE == 'classification':
    flatten = layers.Flatten()(inputs)
    hidden = layers.Dense(32, activation='relu')(flatten)
    output = layers.Dense(10, activation='softmax')(hidden)
else:
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1)(x)

loss_fn = 'sparse_categorical_crossentropy' if MODE == 'classification' else 'mse'
metrics = ['accuracy'] if MODE == 'classification' else ['mae', 'mse']


model = keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=metrics)

from imbal.classification import DatasetWithBatching, generate_weights

sampler = DatasetWithBatching(
    x_train,
    y_train,
    batch_size=512,
    sample_weights=generate_weights(y_train),
)

epochs = 30 if MODE == 'classification' else 150

history = model.fit(sampler, epochs=epochs)

import imbal

if MODE == 'classification':
    imbal.classification.tsne_visualization(model, x_test, y_test, save_figure=SAVE_FIG_NAME)
else:
    imbal.regression.tsne_visualization(model, x_test, y_test, sort='descending', save_figure=SAVE_FIG_NAME)



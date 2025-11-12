import keras
from keras import layers
import numpy as np
from tensorflow.keras.datasets import mnist

import imbal
import shap

LIME_MODE = 'image'
MODE = 'classification'

IMBALANCED = False
HIGH_IMBALANCE = False
SAVE_FIG_NAME = f'tsne-{MODE}-imbalanced-{IMBALANCED}.png'

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[..., np.newaxis].astype(np.float32) / 255.0  # (60000,28,28,1)
x_test = x_test[..., np.newaxis].astype(np.float32) / 255.0    # (10000,28,28,1)

inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(16, (3, 3), activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(
    10,
    activation='softmax'
)(x)

loss_fn = ('sparse_categorical_crossentropy' if HIGH_IMBALANCE == False else 'binary_crossentropy') if MODE == 'classification' else 'mse'
metrics = ['accuracy'] if MODE == 'classification' else ['mae', 'mse']
optimizer = 'adam'

model = keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=metrics)

model.load_weights(f'mnist-trained-{MODE}-model-imbalanced-{IMBALANCED}.weights.h5')

model.summary()

print(x_test.shape)
result = model.predict(np.reshape(x_test[0], (1, 28, 28, 1)))

EXPLAIN_INDEX_START = 20
EXPLAIN_AMOUNT = 10

import tensorflow as tf
background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)].astype(np.float32)
explainer = shap.GradientExplainer(model, background)

for i in range(EXPLAIN_AMOUNT):
    def f(X):
        tmp = X.copy()
        return model(tmp)

    x_ = x_test[i + EXPLAIN_INDEX_START]
    y_ = y_test[i + EXPLAIN_INDEX_START]

    x_sample =x_.reshape(1, 28, 28, 1).astype(np.float32)
    shap_values, indexes = explainer.shap_values(x_sample, ranked_outputs=1)
    vals = shap_values[0]
    # vals = vals / np.max(np.abs(vals))
    # print(x_.shape)
    shap.image_plot(vals.reshape(1, 28, 28, 1), x_.reshape(1, 28, 28, 1).astype(np.float32))


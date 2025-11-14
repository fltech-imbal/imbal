import keras
from keras import layers
import numpy as np
from keras.utils import to_categorical

import imbal
import shap

LIME_MODE = 'image'
MODE = 'classification'

IMBALANCED = False
HIGH_IMBALANCE = False
SAVE_FIG_NAME = f'tsne-{MODE}-imbalanced-{IMBALANCED}.png'

num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)


model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

model.load_weights(f'mnist-trained-{MODE}-model-imbalanced-{IMBALANCED}.weights.h5')

model.summary()

print(x_test.shape)
result = model.predict(np.reshape(x_test[0], (1, 28, 28, 1)))

EXPLAIN_INDEX_START = 20
EXPLAIN_AMOUNT = 5

import tensorflow as tf
# background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)].astype(np.float32)
# explainer = shap.GradientExplainer(model, background)

for i in range(EXPLAIN_AMOUNT):
    # def f(X):
    #     tmp = X.copy()
    #     return model(tmp)
    #
    # x_ = x_test[i + EXPLAIN_INDEX_START]
    # y_ = y_test[i + EXPLAIN_INDEX_START]
    #
    # x_sample =x_.reshape(1, 28, 28, 1).astype(np.float32)
    # shap_values, indexes = explainer.shap_values(x_sample, ranked_outputs=1)
    # vals = shap_values[0]
    # # vals = vals / np.max(np.abs(vals))
    # # print(x_.shape)
    # shap.image_plot(vals.reshape(1, 28, 28, 1), x_.reshape(1, 28, 28, 1).astype(np.float32))


    background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(x_test[i:i+1])

    print('better')
    print(shap_values[0][..., int(y_test[i].argmax())].shape)
    print(x_test[i].shape)
    shap.image_plot(shap_values[0][..., int(y_test[i].argmax())], x_test[i])

    shap_values = np.transpose(shap_values, (0, 1, 2, 3, 4))
    demo_value = x_test[i][np.newaxis, :]

    print('demo')
    print(shap_values.shape)
    print(demo_value.shape)
    shap.image_plot([shap_values], demo_value)


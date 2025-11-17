import keras
from keras import layers
import numpy as np
from skimage.segmentation import slic

import imbal
import shap

LIME_MODE = 'image'
MODE = 'classification'

IMBALANCED = False
HIGH_IMBALANCE = False
SAVE_FIG_NAME = f'tsne-{MODE}-imbalanced-{IMBALANCED}.png'

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
x_combined = x_combined[:percent_index].astype(np.float32)
y_combined = y_combined[:percent_index].astype(np.float32)
num_data = x_combined.shape[0]
split_index = int(num_data * TRAIN_SPLIT)
x_train, x_test = x_combined[:split_index], x_combined[split_index:]
y_train, y_test = y_combined[:split_index], y_combined[split_index:]
print('x_train', x_train.shape)
print('y_train', y_train.shape)
print('x_test', x_test.shape)
print('y_test', y_test.shape)


inputs = keras.Input(shape=(96,96,3))
output = None

x = layers.Conv2D(16, (7, 7), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(32, (7, 7), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
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

model.load_weights(f'stl-10-trained-{MODE}-model-imbalanced-{IMBALANCED}.weights.h5')

model.summary()

print(x_test.shape)
result = model.predict(np.reshape(x_test[0], (1, 96, 96, 3)))

class_labels = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

EXPLAIN_INDEX_START = 80
EXPLAIN_AMOUNT = 5

from matplotlib import pyplot as plt

for i in range(EXPLAIN_AMOUNT):
    def f(X):
        tmp = X.copy()
        return model(tmp)

    x_ = x_test[i + EXPLAIN_INDEX_START]
    y_ = y_test[i + EXPLAIN_INDEX_START]

    imbal.classification.shap_explain_image_sample(
        x_test[i + EXPLAIN_INDEX_START],
        model,
        x_train,
        class_names=class_labels,
        actual_label=y_test[i + EXPLAIN_INDEX_START],
        num_samples=100,
        save_figure=True,
        figure_save_path=f'stl10-explanation-shap-{i}.png',
    )

    imbal.classification.shap_explain_image_sample(
        x_test[i + EXPLAIN_INDEX_START],
        model,
        x_train,
        class_names=class_labels,
        label_to_explain=y_test[i + EXPLAIN_INDEX_START],
        actual_label=y_test[i + EXPLAIN_INDEX_START],
        num_samples=100,
        save_figure=True,
        figure_save_path=f'stl10-explanation-shap-{i}-override.png',
    )


    # background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    #
    # e = shap.DeepExplainer(model, background)
    # shap_values = e.shap_values(np.array([x_test[i + EXPLAIN_INDEX_START]]))
    # shap_values = np.transpose(shap_values[0], (3, 0, 1, 2))
    # print(shap_values.shape)
    #
    # # if label_to_explain is None:
    # #     label_to_explain = int(model.predict(np.array([x_test[i + EXPLAIN_INDEX_START]])).argmax())
    # # shap.image_plot(shap_values[0][..., label_to_explain], x_test[i + EXPLAIN_INDEX_START], show=False)
    #
    # shap.image_plot([x for x in shap_values], x_test[i + EXPLAIN_INDEX_START], show=False)
    #
    # # explanation_label = label_to_explain
    # # if class_names is not None:
    # #     explanation_label = class_names[label_to_explain]
    # # title_string = f'Explanation for "{explanation_label}"'
    # #
    # # if actual_label is not None:
    # #     if class_names is not None:
    # #         actual_label = class_names[actual_label]
    # #     title_string += f' (Actual label: {actual_label})'
    # # plt.suptitle(title_string)
    # plt.show()


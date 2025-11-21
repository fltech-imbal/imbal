import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
import shap

import imbal.backend.explanation
MODE='regression'
IMBALANCED = False
HIGH_IMBALANCE = False
SAVE_FIG_NAME = f'tsne-{MODE}-imbalanced-{IMBALANCED}.png'

from sklearn.datasets import fetch_california_housing

x, y = fetch_california_housing(return_X_y=True)
labels = fetch_california_housing().feature_names

print(labels)

shuffle = np.random.permutation(len(x))
x = x[shuffle]
y = y[shuffle]


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
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(1)(x)

loss_fn = 'mse'
metrics = ['mae', 'mse']
optimizer = keras.optimizers.Adam(learning_rate=2e-4)
model = keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=metrics)

model.load_weights(f'housing-trained-tabular-model-{MODE}.weights.h5')


print(y_test[:20])

EXPLAIN_INDEX_START = 0
EXPLAIN_AMOUNT = 5

for i in range(EXPLAIN_AMOUNT):
    x = x_test[i + EXPLAIN_INDEX_START]
    y = y_test[i + EXPLAIN_INDEX_START]

    # explainer = shap.Explainer(model, x_train)
    # shap_values = explainer(np.expand_dims(x, axis=0))
    #
    # prediction = model.predict(np.expand_dims(x, axis=0))[0]
    # print(prediction)
    # single_class_expl = shap.Explanation(
    #     values=shap_values.values[0],
    #     base_values=shap_values.base_values[0],
    #     data=shap_values.data[0],
    #     feature_names=labels
    # )
    # fig, ax = plt.subplots()
    # shap.plots.waterfall(single_class_expl, show=False)
    # plt.show()
    # fig, ax = plt.subplots()
    # shap.plots.bar(single_class_expl, show=False)
    # plt.show()

    for plot_type in ['bar', 'waterfall']:
        imbal.regression.shap_tabular_explanation(
            x,
            model,
            x_train,
            feature_names=labels,
            plot_type=plot_type
        )



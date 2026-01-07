import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.datasets import fetch_california_housing
import time

MODE = 'decoupled'
FILTER = ''

num_classes = 10
input_shape = (8,)

DATASET_PERCENTAGE = 0.8
TRAIN_SPLIT = 0.8

data = fetch_california_housing()
x_combined, y_combined = data.data, data.target
y_combined = y_combined.reshape(-1,)

print(x_combined.shape)
print(y_combined.shape)

num_data = x_combined.shape[0]
percent_index = int(num_data * DATASET_PERCENTAGE)
# shuffled_indices = np.random.permutation(len(x_combined))[:percent_index]
# x_combined = x_combined[shuffled_indices].astype(np.float32)
# y_combined = y_combined[shuffled_indices].astype(np.float32)
num_data = x_combined.shape[0]
split_index = int(num_data * TRAIN_SPLIT)
x_train, x_test = x_combined[:split_index], x_combined[split_index:]
y_train, y_test = y_combined[:split_index], y_combined[split_index:]
print('x_train', x_train.shape)
print('y_train',y_train.shape)
print('x_test',x_test.shape)
print('y_test',y_test.shape)

print(y_train.shape)
print(x_test.shape)

class_split = []
for i in range(num_classes):
    class_split.append(len(y_train[y_train == i]))
print('distribution', class_split)

# y_train = to_categorical(y_train, num_classes if FILTER != 'binary' else 2)
# y_test = to_categorical(y_test, num_classes if FILTER != 'binary' else 2)

inputs = keras.Input(shape=input_shape)
x = layers.Dense(32, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(1)(x)

model = keras.Model(inputs=inputs, outputs=output)

model.summary()

import imbal

batch_size = 512
epochs = 100

print('number of layers', len(model.layers))

auc = keras.metrics.AUC(multi_label=True)

parameters = imbal.classification.compile_parameters(
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    metrics=["mse"]
)

start = time.time()
if MODE == 'decoupled':
    imbal.classification.decoupled_fit(
        model,
        x_train,
        y_train,
        compile_parameters=parameters,
        epochs=epochs,
        batch_size=batch_size,
        representation_layer_index=-3
    )
else:
    model.compile(**parameters.to_dict())
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs
    )

end = time.time()

print('EXECUTION TIME:', end - start)

print('Evaluating model...')
model.evaluate(x_test, y_test)

predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)

print(predictions)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# x






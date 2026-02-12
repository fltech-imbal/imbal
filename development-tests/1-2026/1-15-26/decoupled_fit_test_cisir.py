import keras
from tensorflow.keras import layers
import numpy as np
import time
from matplotlib import pyplot as plt
import tensorflow as tf

MODE = 'decoupled'
FILTER = ''
AE = True

num_classes = 10

DATASET_PERCENTAGE = 0.8
TRAIN_SPLIT = 0.8

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

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
for i in range(num_classes):
    x_train_filter.append(x_train[y_train == i][:(10-i)*500])
    y_train_filter.append(y_train[y_train == i][:(10-i)*500])
    x_test_filter.append(x_test[y_test == i][:(10-i)*90])
    y_test_filter.append(y_test[y_test == i][:(10-i)*90])

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

x_train = x_train / 255
x_test = x_test / 255

print('image', x_train[0].shape)
print(x_train[0].min(), x_train[0].max())
plt.imshow(x_train[0])
plt.show()

class_split = []
for i in range(num_classes):
    class_split.append(len(y_train[y_train == i]))
print('distribution', class_split)

print("SHAPES")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

inputs = keras.Input(shape=(28,28,1))
x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same')(inputs)
x = layers.LayerNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(x)
x = layers.LayerNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
x = layers.LayerNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(1)(x)

model = keras.Model(inputs=inputs, outputs=output)

model.summary()

import imbal

batch_size = 512
epochs = 60

print('number of layers', len(model.layers))

auc = keras.metrics.AUC(multi_label=True)
f1 = tf.keras.metrics.F1Score()

parameters = imbal.regression.wrap_model_compile_parameters(
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    metrics=["mse"]
)

REPRESENTATION_LAYER_INDEX = -6

start = time.time()
if MODE == 'decoupled':
    bandwidth = imbal.regression.fit_kde(
        y_train,
        bin_count=64
    )

    densities = imbal.regression.get_sample_densities(
        y_train,
        bandwidth
    )

    imbal.regression.rRT_fit(
        model,
        x_train,
        y_train,
        compile_parameters=parameters,
        sample_densities=densities,
        epochs=epochs,
        batch_size=batch_size,
        generate_decoder_branch=AE,
        representation_layer_index=REPRESENTATION_LAYER_INDEX,
    )

elif MODE == 'balanced':
    print('fitting kde...')
    # bandwidth = imbal.regression.fit_kde(y_train)
    bandwidth = 0.2
    print('generating densities...')
    densities = imbal.regression.get_sample_densities(
        y_train,
        bandwidth
    )
    print('performing balanced fit...')
    imbal.regression.balanced_fit(
        model,
        x_train,
        y_train,
        sample_densities=densities,
        compile_parameters=parameters,
        epochs=epochs,
        batch_size=batch_size,
        generate_decoder_branch=AE,
        representation_layer_index=REPRESENTATION_LAYER_INDEX
    )
else:
    if AE:
        extended_model, _ = imbal.util.backend.fit.generate_decoder_branch(model, REPRESENTATION_LAYER_INDEX)
        extended_parameters = imbal.regression.wrap_model_compile_parameters(
            loss=["mse", 'mse'],
            optimizer=keras.optimizers.Adam(learning_rate=2e-5),
            metrics=[["mse"], ['mse']]
        )
        extended_model.compile(**extended_parameters.to_dict())
        extended_model.fit(
            x_train,
            [y_train, x_train],
            batch_size=batch_size,
            epochs=epochs
        )
        model.compile(**parameters.to_dict())
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

import matplotlib.pyplot as plt

BIN_COUNT = 64
kde_bandwidth = imbal.regression.fit_kde(y_combined, bin_count=BIN_COUNT)
imbal.regression.plot_kde_1d(
    y_combined,
    kde_bandwidth,
    bin_count=BIN_COUNT,
    save_figure='sep-c-data-distribution.png'
)

predictions = predictions.reshape(-1, 1)
print(y_test)
print(y_test.shape)
print(predictions)
print(predictions.shape)
print(np.min(predictions), np.max(predictions))
plt.scatter(y_test, predictions)
plt.plot([-1, 10],[-1, 10], linestyle='--', color='red')
plt.xlabel('Data label')
plt.ylabel('Prediction')
plt.xlim(-1, 10)
plt.ylim(-1, 10)
plt.savefig(f'fit-comparison-{MODE}-ae-{AE}.png')
plt.show()

plt.scatter(y_test, predictions)
plt.plot([-10, 10],[-10, 10], linestyle='--', color='red')
plt.xlabel('Data label')
plt.ylabel('Prediction')
plt.xlim(-2, 2)
plt.ylim(np.min(predictions)*1.05, np.max(predictions)*1.05)
plt.show()


imbal.regression.tsne_visualization(
    model,
    x_test,
    y_test,
    save_figure=f'tsne_visualization-{MODE}-ae-{AE}.png',
)


predictions = predictions.reshape(-1,)

common_range = (-1, 1)

mask = y_test == 0
rare_mask = y_test == 9

common_predictions = predictions[mask]
rare_predictions = predictions[rare_mask]

common_labels = y_test[mask]
rare_labels = y_test[rare_mask]

print(rare_labels)
print(rare_predictions)

mse_common = np.mean(np.square(common_predictions - common_labels))
mse_rare = np.mean(np.square(rare_predictions - rare_labels))

print('common', f'{mse_common:.5f}')
print('rare', f'{mse_rare:.5f}')





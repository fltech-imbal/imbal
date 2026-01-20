import keras
from tensorflow.keras import layers
import numpy as np
import time
import os, csv

MODE = ''
FILTER = ''
AE = False



num_classes = 10
REPRESENTATION_LAYER_INDEX = -6
DATASET_PERCENTAGE = 0.8
TRAIN_SPLIT = 0.8

def read_csv_to_list_of_lists(filepath):
    data = []
    with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            data.append(row)
    return data

PATH_START = '/mnt/c/Users/tommy/PycharmProjects/DrChanWorkPlayground'
print(os.getcwd())

def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0

safe_float_vectorized = np.vectorize(safe_float)


# from sklearn.preprocessing import StandardScaler
# data = np.array(read_csv_to_list_of_lists(f'{PATH_START}/tutorials/data/SARCOS/sarcos_inv_training.csv'))
# print(data.shape)
# y_combined = data[1:, -1].astype(float)
# data = safe_float_vectorized(data).astype(float)
# scaler = StandardScaler()
# NUM_FEATURES = data.shape[1] - 1
# x_combined = data[1:, :NUM_FEATURES].astype(float)
# # x_combined = scaler.fit_transform(x_combined)


from sklearn.preprocessing import StandardScaler
data = np.array(read_csv_to_list_of_lists(f'{PATH_START}/tutorials/data/SEP-C/sep_10mev_training.csv'))
print(data.shape)
data = safe_float_vectorized(data[1:]).astype(float)
NUM_FEATURES = 22
y_combined = data[:, NUM_FEATURES].astype(float)
scaler = StandardScaler()
x_combined = data[:, :NUM_FEATURES].astype(float)
x_combined = scaler.fit_transform(x_combined)


# from sklearn.preprocessing import StandardScaler
# data = np.array(read_csv_to_list_of_lists(f'{PATH_START}/tutorials/data/SEP-EC/training/sep_event_1_filled_ie_trim.csv'))[1:]
# for i in range(43):
#     if os.path.exists(f'{PATH_START}/tutorials/data/SEP-EC/training/sep_event_{i+2}_filled_ie_trim.csv'):
#         data = np.concatenate([data, read_csv_to_list_of_lists(f'{PATH_START}/tutorials/data/SEP-EC/training/sep_event_{i+2}_filled_ie_trim.csv')[1:]])
# print(data.shape)
# data = safe_float_vectorized(data).astype(float)
# y_combined = data[:, 182].astype(float)
# scaler = StandardScaler()
# NUM_FEATURES = 182
# x_combined = data[:, :NUM_FEATURES].astype(float)
# x_combined = scaler.fit_transform(x_combined)




print(x_combined.shape)
print(y_combined.shape)

num_data = x_combined.shape[0]
percent_index = int(num_data * DATASET_PERCENTAGE)
shuffled_indices = np.random.RandomState(seed=0).permutation(len(x_combined))[:percent_index]
x_combined = x_combined[shuffled_indices].astype(np.float32)
y_combined = y_combined[shuffled_indices].astype(np.float32)
num_data = x_combined.shape[0]
split_index = int(num_data * TRAIN_SPLIT)
x_train, x_test = x_combined[:split_index], x_combined[split_index:]
y_train, y_test = y_combined[:split_index], y_combined[split_index:]
import math
y_train = (y_train >= math.log(10)).astype(int)
y_test = (y_test >= math.log(10)).astype(int)
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

input_shape = (NUM_FEATURES,)

inputs = keras.Input(shape=input_shape)
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dense(16, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=output)

model.summary()

import imbal

batch_size = 512
epochs = 2000

print('number of layers', len(model.layers))

auc = keras.metrics.AUC(multi_label=True)

parameters = imbal.classification.wrap_model_compile_parameters(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    metrics=["accuracy"]
)
BIN_COUNT=64

kde_bandwidth = imbal.regression.fit_kde(
    y_train,
    bin_count=BIN_COUNT
)
densities = imbal.regression.get_sample_densities(
    y_train,
    kde_bandwidth,
)

start = time.time()
if MODE == 'decoupled':
    bandwidth = imbal.regression.fit_kde(
        y_train,
        bin_count=BIN_COUNT
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
    bandwidth = imbal.regression.fit_kde(
        y_train,
        bin_count=BIN_COUNT
    )
    densities = imbal.regression.get_sample_densities(
        y_train,
        bandwidth
    )

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
        extended_parameters = imbal.classification.wrap_model_compile_parameters(
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

kde_bandwidth = imbal.regression.fit_kde(y_combined, bin_count=BIN_COUNT)
imbal.regression.plot_kde_1d(
    y_combined,
    kde_bandwidth,
    bin_count=BIN_COUNT,
    save_figure='sep-ec-kde-curve.png'
)


# plt.scatter(y_test, predictions)
# plt.plot([-10, 10],[-10, 10], linestyle='--', color='red')
# plt.xlabel('Data label')
# plt.ylabel('Prediction')
# plt.xlim(-2, 2)
# plt.ylim(-2, 2)
# plt.savefig(f'fit-comparison-{MODE}-ae-{AE}.png')
# plt.show()
#
# plt.scatter(y_test, predictions)
# plt.plot([-10, 10],[-10, 10], linestyle='--', color='red')
# plt.xlabel('Data label')
# plt.ylabel('Prediction')
# plt.xlim(-2, 2)
# plt.ylim(np.min(predictions)*1.05, np.max(predictions)*1.05)
# plt.show()


imbal.classification.tsne_visualization(
    model,
    x_test,
    y_test,
    save_figure=f'tsne_visualization-{MODE}-ae-{AE}.png',
)


predictions = predictions.reshape(-1,)

# mask = (y_test >= common_range[0]) & (y_test <= common_range[1])
# rare_mask = (y_test < common_range[0]) | (y_test > common_range[1])
# common_predictions = predictions[mask]
# rare_predictions = predictions[rare_mask]
#
# common_labels = y_test[mask]
# rare_labels = y_test[rare_mask]
#
# mse_common = np.mean(np.square(common_predictions - common_labels))
# mse_rare = np.mean(np.square(rare_predictions - rare_labels))
#
# print('common', f'{mse_common:.5f}')
# print('rare', f'{mse_rare:.5f}')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_test_labels = y_test
predictions_labels = (predictions >= 0.5).astype(int)

cm = confusion_matrix(y_test_labels, predictions_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot()
plt.savefig(f'confusion-matrix-{MODE}-ae-{AE}.png')
plt.show()

import tensorflow as tf
f1_score = tf.keras.metrics.F1Score(threshold=0.5)
f1_score.update_state(y_test_labels.reshape(-1, 1), predictions.reshape(-1, 1))

auroc = tf.keras.metrics.AUC(num_thresholds=2000)
print(y_test_labels.shape)
print(y_test_labels[:20])
print(predictions.shape)
auroc.update_state(y_test_labels, predictions)
print(auroc.result())

print(np.max(predictions[y_test_labels == 0]))
print(predictions[y_test_labels == 1][:20])

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_scores = predictions

fpr, tpr, thresholds = roc_curve(y_test_labels, y_scores, drop_intermediate=False)
roc_auc = auc(fpr, tpr)
print("sklearn AUROC:", roc_auc)

plt.figure(figsize=(7, 6))
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

points = np.array([fpr, tpr]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm_thresholds = thresholds

lc = LineCollection(
    segments,
    cmap='viridis',
    norm= plt.Normalize(vmin=0, vmax=1)
)
lc.set_array(norm_thresholds)
lc.set_linewidth(2)

fig, ax = plt.subplots(figsize=(7, 6))
ax.add_collection(lc)
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
cbar = plt.colorbar(lc, ax=ax)
cbar.set_label("Decision Threshold")

plt.savefig(f'roc-curve-{MODE}-ae-{AE}.png')
plt.show()

print(f1_score.result())





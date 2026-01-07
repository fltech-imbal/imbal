import keras
from tensorflow.keras import layers
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt

MODE = ''
FILTER = 'binary'
BALANCE = 'low'

AE = 'hand'

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
for i in range(num_classes):
        if i == 6:
            break
        if i == 0:
            x_train_filter.append(x_train[y_train == i])
            y_train_filter.append(y_train[y_train == i])
            x_test_filter.append(x_test[y_test == i])
            y_test_filter.append(y_test[y_test == i])
        else:
            if i < 5:
                continue
            x_train_filter.append(x_train[y_train == i][:40 if BALANCE == 'high' else 200])
            y_train_filter.append(np.ones(y_train[y_train == i][:40 if BALANCE == 'high' else 200].shape))
            x_test_filter.append(x_test[y_test == i][:10 if BALANCE == 'high' else 50])
            y_test_filter.append(np.ones(y_test[y_test == i][:10 if BALANCE == 'high' else 50].shape))

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

# x_train = tf.image.rgb_to_grayscale(x_train).numpy()
# x_test = tf.image.rgb_to_grayscale(x_test).numpy()

class_split = []
for i in range(num_classes):
    class_split.append(len(y_train[y_train == i]))
print('distribution', class_split)

y_train = to_categorical(y_train, num_classes if FILTER != 'binary' else 2)
y_test = to_categorical(y_test, num_classes if FILTER != 'binary' else 2)

print(x_test.shape)
print(y_train.shape)

rares = np.where(np.argmax(y_test, axis=1) == 1)[0]
print(rares)

"""
Build model
"""

inputs = keras.Input(shape=input_shape)

x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same')(inputs)
x = layers.LayerNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(x)
x = layers.LayerNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
x = layers.LayerNormalization()(x)
x = layers.Activation('relu')(x)

latent = layers.Flatten()(x)
x = layers.Dropout(0.4)(latent)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(num_classes if FILTER != 'binary' else 2, activation='softmax')(x)

# Hand-crafted AE
if AE == 'hand':
    y = layers.Reshape((4, 4, 64))(latent)

    y = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(y)
    y = layers.LayerNormalization()(y)
    y = layers.Activation('relu')(y)

    y = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(y)
    y = layers.LayerNormalization()(y)
    y = layers.Activation('relu')(y)

    y = layers.Conv2DTranspose(3, (3, 3), strides=(2, 2),  padding='same')(y)
    y = layers.LayerNormalization()(y)
    input_recon = layers.Activation('relu')(y)
elif AE == 'mirror':
    # True mirror AE
    y = layers.Reshape((4, 4, 64))(latent)

    y = layers.Activation('relu')(y)
    y = layers.LayerNormalization()(y)
    y = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(y)

    y = layers.Activation('relu')(y)
    y = layers.LayerNormalization()(y)
    y = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(y)

    y = layers.Activation('relu')(y)
    y = layers.LayerNormalization()(y)
    input_recon = layers.Conv2DTranspose(3, (3, 3), strides=(2, 2),  padding='same')(y)


model = keras.Model(inputs=inputs, outputs=[output, input_recon])

model.summary()

for layer in model.layers:
    if isinstance(layer, layers.InputLayer):
        continue
    print(layer.name, layer.input.shape, layer.output.shape)

import imbal

batch_size = 512
epochs = 600

print('number of layers', len(model.layers))

auc = keras.metrics.AUC(multi_label=True)
f1 = tf.keras.metrics.F1Score()

parameters = imbal.classification.wrap_model_compile_parameters(
    loss=["categorical_crossentropy", 'mse'],
    optimizer=keras.optimizers.Adam(learning_rate=2e-4),
    metrics=[["accuracy", f1, auc], ['mse']]
)

model.compile(**parameters.to_dict())

if MODE == 'balanced':
    sample_weights = imbal.classification.generate_sample_weights(y_train)
else:
    sample_weights = np.ones(y_train.shape[0]).reshape(-1)

image_weights = np.repeat(sample_weights, 32*32).reshape(y_train.shape[0], 32, 32)

model.fit(
    x_train,
    [y_train, x_train],
    sample_weight=[sample_weights, image_weights],
    batch_size=batch_size,
    epochs=epochs
)

predictions = model.predict(x_test)

plt.imshow(x_test[0])
plt.show()
plt.imshow(predictions[1][0])
plt.show()
for i in range(5):
    plt.imshow(x_test[rares[i]])
    plt.show()
    plt.imshow(predictions[1][rares[i]])
    plt.show()

y_test_labels = np.argmax(y_test, axis=1)

predictions = predictions[0]
predictions_labels = np.argmax(predictions, axis=1)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test_labels, predictions_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Airplane", "Dog"])
disp.plot()
plt.savefig(f'confusion-matrix-{MODE}-{BALANCE}-{AE}.png')
plt.show()

# imbal.classification.tsne_visualization(
#     model,
#     x_test,

#     y_test_labels,
#     save_figure=f'tsne_visualization-{MODE}-{BALANCE}.png',
# )
import tensorflow as tf
f1_score = tf.keras.metrics.F1Score()
f1_score.update_state(y_test, predictions)


auroc = tf.keras.metrics.AUC(num_thresholds=2000)
auroc.update_state(y_test_labels, predictions[:, 1].reshape(-1, 1))
print(auroc.result())

print(np.max(predictions[y_test_labels == 0][:, 1]))
print(predictions[y_test_labels == 1][:20, 1])

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_scores = predictions[:, 1]

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

plt.savefig(f'roc-curve-{MODE}-{BALANCE}-{AE}.png')
plt.show()

print(f1_score.result())



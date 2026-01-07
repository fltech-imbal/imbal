import keras
from tensorflow.keras import layers
import numpy as np
from keras.utils import to_categorical
import time
import tensorflow as tf

MODE = 'decoupled'
FILTER = 'binary'

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

class_split = []
for i in range(num_classes):
    class_split.append(len(y_train[y_train == i]))
print(class_split)

x_train_filter = []
y_train_filter = []
x_test_filter = []
y_test_filter = []
print(np.tile(x_train[y_train==0], [1, 1, 1, 1]).shape)
for i in range(num_classes):
    if FILTER == 'binary':
        if i == 6:
            break
        if i == 0:
            x_train_filter.append(np.tile(x_train[y_train == i], [1, 1, 1, 1]))
            y_train_filter.append(np.tile(y_train[y_train == i], 1))
            x_test_filter.append(np.tile(x_test[y_test == i], [1, 1, 1, 1]))
            y_test_filter.append(np.tile(y_test[y_test == i], 1))
        else:
            if i < 5:
                continue
            x_train_filter.append(np.ones(x_train[y_train == i][:200].shape))
            y_train_filter.append(np.ones(y_train[y_train == i][:200].shape))
            x_test_filter.append(np.ones(x_test[y_test == i][:50].shape))
            y_test_filter.append(np.ones(y_test[y_test == i][:50].shape))
    elif i < 1 or FILTER != 'imbalanced':
        x_train_filter.append(np.tile(x_train[y_train==i], [1, 1, 1, 1]))
        y_train_filter.append(np.tile(y_train[y_train==i], 1))
        x_test_filter.append(np.tile(x_test[y_test==i], [1, 1, 1, 1]))
        y_test_filter.append(np.tile(y_test[y_test==i], 1))
    else:
        x_train_filter.append(x_train[y_train == i][:10])
        y_train_filter.append(y_train[y_train == i][:10])
        x_test_filter.append(x_test[y_test == i][:10])
        y_test_filter.append(y_test[y_test == i][:10])

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

print(y_train.shape)
print(x_test.shape)

class_split = []
for i in range(num_classes):
    class_split.append(len(y_train[y_train == i]))
print('distribution', class_split)

y_train = to_categorical(y_train, num_classes if FILTER != 'binary' else 2)
y_test = to_categorical(y_test, num_classes if FILTER != 'binary' else 2)

inputs = keras.Input(shape=input_shape)
x = layers.Conv2D(16, (3, 3), strides=(2, 2))(inputs)
x = layers.LayerNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(32, (3, 3), strides=(2, 2))(x)
x = layers.LayerNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(64, (3, 3), strides=(2, 2))(x)
x = layers.LayerNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(num_classes if FILTER != 'binary' else 2, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=output)

model.summary()

import imbal

batch_size = 512
epochs = 30

print('number of layers', len(model.layers))

auc = keras.metrics.AUC(multi_label=True)
f1 = tf.keras.metrics.F1Score()

parameters = imbal.classification.wrap_model_compile_parameters(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    metrics=["accuracy", f1, auc]
)

start = time.time()
if MODE == 'decoupled':
    imbal.classification.decoupled_fit(
        model,
        x_train,
        y_train,
        compile_parameters=parameters,
        epochs=epochs,
        batch_size=batch_size
    )

elif MODE == 'balanced':
    imbal.classification.balanced_fit(
        model,
        x_train,
        y_train,
        compile_parameters=parameters,
        epochs=epochs,
        batch_size=batch_size
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
y_test_labels = np.argmax(y_test, axis=1)
print(y_test_labels.shape)
print(y_test_labels)
print(np.unique(y_test_labels, return_counts=True))

predictions = model.predict(x_test)
predictions_labels = np.argmax(predictions, axis=1)

print(predictions_labels)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test_labels, predictions_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Airplane", "Dog"])
disp.plot()
plt.savefig(f'confusion-matrix-{MODE}.png')
plt.show()

print(y_test_labels.shape)
imbal.classification.tsne_visualization(
    model,
    x_test,
    y_test_labels,
    save_figure=f'tsne_visualization-{MODE}.png',
)
import tensorflow as tf
f1_score = tf.keras.metrics.F1Score()
f1_score.update_state(y_test, predictions)
print(y_test[:20])
print(predictions[:20])
print(f1_score.result())

auroc = tf.keras.metrics.AUC(num_thresholds=2000)
print(y_test_labels[:20])
print(predictions[:, 1].reshape(-1, 1)[:20])
auroc.update_state(y_test_labels, predictions[:, 1].reshape(-1, 1))
print(auroc.result())

print(np.max(predictions[y_test_labels == 0][:, 1]))
print(predictions[y_test_labels == 1][:20, 1])

f1_score = tf.keras.metrics.F1Score(threshold=np.max(predictions[y_test_labels == 0][:, 1])+.005)
f1_score.update_state(y_test, predictions)
print('adjusted f1')
print(f1_score.result())

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_scores = predictions[:, 1]

fpr, tpr, thresholds = roc_curve(y_test_labels, y_scores)

roc_auc = auc(fpr, tpr)
print("sklearn AUROC:", roc_auc)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)  # baseline diagonal

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUROC Curve")
plt.legend(loc="lower right")
plt.grid(True)

plt.savefig(f'roc-curve-{MODE}.png', dpi=300)
plt.show()
import keras
from tensorflow.keras import layers
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
import time

MODE = 'decoupled'
BALANCE = 'low'

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

class_split = []
for i in range(num_classes):
    class_split.append(len(y_train[y_train == i]))
print('distribution', class_split)

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

print(x_test.shape)
print(y_train.shape)

rares = np.where(np.argmax(y_test, axis=1) == 1)[0]
print(rares)

from imbal.util.backend import positive_model_layer_index
def generate_ae_branch(
    model,
    representation_layer_index=-2
):
    representation_layer_index = positive_model_layer_index(model, representation_layer_index)
    reverse_model = model.layers[:representation_layer_index][::-1]

    # Determine AE blocks
    ae_blocks = []
    last_block_end_index = 0
    for index, layer in enumerate(reverse_model):
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            ae_blocks.append(reverse_model[last_block_end_index:index+1][::-1])
            last_block_end_index = index+1
    # Exclude input layer
    if last_block_end_index != len(reverse_model) - 1:
        ae_blocks.append(reverse_model[last_block_end_index:-1][::-1])

    # Perform per-layer conversions (i.e. Conv2D to Conv2DTranspose)
    # within each block
    ae_branch_blocks = []
    for block_index, block in enumerate(ae_blocks):
        current_input_shape = block[-1].output.shape[1:]
        current_ae_block = []
        reshape_layer = keras.layers.Reshape(current_input_shape, name=f'imbal_auto_generated_ae_safeguard_reshape_block_{block_index}')
        current_ae_block.append(reshape_layer)
        print('----- BLOCK -----') # Debug, delete later
        for layer_index, layer in enumerate(block):
            new_layer = None
            config = layer.get_config()
            config['name'] = f'imbal_auto_generated_ae_block_{block_index}_layer_{layer_index}'
            if isinstance(layer, keras.layers.Conv2D):
                layer_shape_change = layer.input.shape[-1] / layer.output.shape[-1]
                config.pop('groups', None)
                config['filters'] = round(config['filters'] * layer_shape_change)
                new_layer = keras.layers.Conv2DTranspose(**config)
            if isinstance(layer, keras.layers.Conv2DTranspose):
                layer_shape_change = layer.input.shape[-1] / layer.output.shape[-1]
                config['filters'] = round(config['filters'] * layer_shape_change)
                new_layer = keras.layers.Conv2D(**config)
            if isinstance(layer, keras.layers.Dense):
                units = layer.input.shape[-1]
                config['units'] = units
                new_layer = keras.layers.Dense(**config)

            # Failsafe for non-trainable layers
            elif not (hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer')):
                new_layer = type(layer).from_config(config)

            # Raise exception if layer could not be converted
            if new_layer is None:
                raise RuntimeError(f'Unable to perform AE conversion of layer {layer}')

            print(f'\t{new_layer}') # Debug, delete later
            print(f'\t\t{new_layer.get_config()}') # Debug, delete later

            current_ae_block.append(new_layer)
        ae_branch_blocks.append(current_ae_block)

    # For better results, last block should only be made on trainable layers (activation
    # and normalization layers can sometimes prevent reaching the goal reconstruction)
    refined_last_block = []
    for layer in ae_branch_blocks[-1]:
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            refined_last_block.append(layer)
    ae_branch_blocks[-1] = refined_last_block

    print('\n----- AE CONVERSION -----\n')  # Debug, delete later
    for block in ae_branch_blocks:
        print('----- BLOCK -----')
        for layer in block:
            print(f'\t{layer}, {layer.get_config()}')  # Debug, delete later

    # Connect final layer structure
    ae_layer_list = [layer for block in ae_branch_blocks for layer in block]
    last_layer = model.layers[representation_layer_index]
    for layer in ae_layer_list:
        layer(last_layer.output)
        last_layer = layer

    ae_extended_model = keras.models.Model(inputs=model.inputs, outputs=model.outputs + [ae_layer_list[-1].output])
    ae_extended_model.summary() # Debug, delete layer

    return ae_extended_model, ae_layer_list

"""
Build model
"""

inputs = keras.Input(shape=input_shape)
x = layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same')(inputs)
x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same')(inputs)
x = layers.Activation('relu')(x)

x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(x)
x = layers.Activation('relu')(x)

x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
x = layers.Activation('relu')(x)

latent = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(latent)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(2, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=output)

model.summary()

extended_model, ae_branch_layers = generate_ae_branch(model, representation_layer_index=-4)

keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
keras.utils.plot_model(extended_model, to_file='extended-model.png',show_shapes=True, show_layer_names=True)


# for layer in model.layers:
#     if isinstance(layer, layers.InputLayer):
#         continue
#     print(layer.name, layer.input.shape, layer.output.shape)

# model = extended_model

import imbal

batch_size = 512
epochs = 600

print('number of layers', len(model.layers))

auc = keras.metrics.AUC(multi_label=True)
f1 = tf.keras.metrics.F1Score()

parameters = imbal.classification.wrap_model_compile_parameters(
    loss=["categorical_crossentropy"],
    optimizer=keras.optimizers.Adam(learning_rate=2e-4),
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
        batch_size=batch_size,
        representation_layer_index=-4,
        generate_decoder_branch=True
    )

elif MODE == 'balanced':
    imbal.classification.balanced_fit(
        model,
        x_train,
        y_train,
        compile_parameters=parameters,
        epochs=epochs,
        batch_size=batch_size,
        generate_decoder_branch=True,
        representation_layer_index=-4
    )
else:
    model.compile(**parameters.to_dict())
    model.fit(
        x_train,
        [y_train, x_train],
        batch_size=batch_size,
        epochs=epochs
    )

end = time.time()
print('EXECUTION TIME:', end - start)
predictions = model.predict(x_test)
print(predictions.shape)
model.evaluate(x_test, y_test)

y_test_labels = np.argmax(y_test, axis=1)

predictions_labels = np.argmax(predictions, axis=1)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test_labels, predictions_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Airplane", "Dog"])
disp.plot()
plt.savefig(f'confusion-matrix-{MODE}-{BALANCE}-ae.png')
plt.show()

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

plt.savefig(f'roc-curve-{MODE}-{BALANCE}-ae.png')
plt.show()

print(f1_score.result())



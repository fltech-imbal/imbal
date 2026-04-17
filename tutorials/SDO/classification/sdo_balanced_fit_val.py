"""
Import packages
"""
import imbal
import os
import numpy as np
from PIL import Image
from keras import layers, optimizers, callbacks, metrics

"""
Load data
"""
SDO_DATA_PATH = '../../data/SDOBenchmark' # Ensure data is located at this path

def load_sdo_data(data_path):
    # Load labels (log peak flux)
    with open(os.path.join(data_path, 'log_peak_flux.txt'), 'r') as file:
        contents = file.read().strip()
        loaded_data_fluxes = np.array([float(x) for x in contents.split('\n')])

    # Load images (10 images per sample, 256x256 per image)
    loaded_images = np.zeros((len(loaded_data_fluxes), 128, 128, 1), dtype=np.float32)
    for i in range(len(loaded_data_fluxes)):
        print(f'Loading SDO samples [{i+1}/{len(loaded_data_fluxes)}]', end='\r')
        image_list = Image.open(os.path.join(data_path, f'sdo_subset_sample_{i}.jpg')).convert('L')
        stacked_images = np.array(image_list).reshape(128, 128, 1) # Images stacked along channels
        loaded_images[i] = stacked_images / 255.0 # Normalize black and white pixel values from 0 to 1

    print(f'\n{len(loaded_data_fluxes)} data samples loaded successfully')
    return loaded_images, loaded_data_fluxes

# Load train and test data via function defined above
x_train, y_train = load_sdo_data(os.path.join(SDO_DATA_PATH, 'training'))
x_test, y_test = load_sdo_data(os.path.join(SDO_DATA_PATH, 'test'))
y_train = (y_train > -4).astype(np.int32)
y_test = (y_test > -4).astype(np.int32)

print(
    f'Loaded data with the following shapes:\n'
    f'\tx_train: {x_train.shape}\n'
    f'\ty_train: {y_train.shape}\n'
    f'\tx_test: {x_test.shape}\n'
    f'\ty_test {y_test.shape}'
)

"""
Build model
"""
def build_simple_cnn():
    input_layer = layers.Input((128, 128, 1))
    x = layers.Conv2D(8, 3, activation='relu', padding='same')(input_layer)
    x = layers.Conv2D(8, 3, activation='relu', padding='same', strides=(2, 2))(x)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(16, 3, activation='relu', padding='same', strides=(2, 2))(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same', strides=(2, 2))(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Flatten()(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)

    model = imbal.classification.Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    return model

model = build_simple_cnn()

"""
Create validation split
"""

(x_train, y_train), (x_val, y_val) =  imbal.classification.split(x_train, y_train, test_size=0.1)

"""
Compile and train model
"""
LEARNING_RATE = 2e-4
BATCH_SIZE = 256
PATIENCE = 10

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', metrics.F1Score(threshold=0.5)],
)

history = model.balanced_fit(
    x_train,
    y_train.reshape(-1, 1),
    validation_data=(x_val, y_val.reshape(-1, 1)),
    epochs=500,
    batch_size=BATCH_SIZE,
    class_weight=[[0.1, 0.9], [0.2, 0.8], [0.3, 0.4], [0.4, 0.4], [0.5, 0.5]],
    stratify_batches=True, # Ensure all batches have a similar data distribution
    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)]
)

print(f'Fit stopped after {len(history.history["loss"])} epochs')
print(f'Restored weights from epoch {len(history.history["loss"]) - PATIENCE}')

model.evaluate(x_test, y_test.reshape(-1, 1))

"""
Data and results visualization
"""
KDE_BIN_COUNT=32

test_rare_mask = y_test == 1
test_frequent_mask = ~test_rare_mask
print('Number of test samples with log10 flux < -4:', np.sum(test_frequent_mask))
print('Number of test samples with log10 flux >= -4:', np.sum(test_rare_mask))

# Predict on test data
test_predictions = []
for i in range(0, len(x_test), BATCH_SIZE):
    batch = x_test[i:i+BATCH_SIZE]
    test_predictions.append(model.predict(batch))
test_predictions = np.concatenate(test_predictions, axis=0)
test_predictions = test_predictions.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Calculate metrics
hss = imbal.metrics.HeikdeSkillScore(threshold=0.5)
hss.update_state(y_test, test_predictions)

f1 = metrics.F1Score(threshold=0.5)
f1.update_state(y_test, test_predictions)

print(
    f'Heikde Skill Score: {hss.result()[0]:.4f}\n'
    f'F1 Score: {f1.result()[0]:.4f}\n'
)

imbal.classification.plot_confusion_matrix(
    y_test,
    test_predictions,
    save_figure='sample-sdo-balanced-fit-val-confusion-matrix.png'
)
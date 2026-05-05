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
Compile and train model
"""
LEARNING_RATE = 5e-5
EPOCHS = 20
BATCH_SIZE = 256

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', metrics.F1Score(threshold=0.5)],
)

model.balanced_fit(
    x_train,
    y_train.reshape(-1, 1),
    # class_weight=[[0.9, 0.1,], [0.6, 0.4], [0.5, 0.5]], # Uncomment to use varying class weights
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    stratify_batches=True # Ensure all batches have a similar data distribution
)

model.evaluate(x_test, y_test.reshape(-1, 1))
test_predictions = model.predict(x_test)

rounded_predictions = np.round(test_predictions).astype(np.int32).reshape(-1)

true_positive_mask = (y_test == 1) & (rounded_predictions == 1)
false_positive_mask = (y_test == 0) & (rounded_predictions == 1)
false_negative_mask = (y_test == 1) & (rounded_predictions == 0)

true_positives = x_test[true_positive_mask]
false_positives = x_test[false_positive_mask]
false_negatives = x_test[false_negative_mask]

imbal.classification.plot_confusion_matrix(
    y_test,
    test_predictions,
    save_figure='classification-explanation-confusion-matrix.png'
)

"""
SHAP
"""
if len(true_positives) > 0:
    imbal.classification.shap_explain_image_sample(
        true_positives[0],
        model,
        x_train,
        class_names=["Log pf <= -4", "Log pf > -4"],
        actual_label=1,
        save_figure='shap-classification-true-positive-explanation.png'
    )

if len(false_positives) > 0:
    imbal.classification.shap_explain_image_sample(
        false_positives[0],
        model,
        x_train,
        class_names=["Log pf <= -4", "Log pf > -4"],
        actual_label=0,
        save_figure='shap-classification-false-positive-explanation.png'
    )

if len(false_negatives) > 0:
    imbal.classification.shap_explain_image_sample(
        false_negatives[0],
        model,
        x_train,
        class_names=["Log pf <= -4", "Log pf > -4"],
        actual_label=1,
        save_figure='shap-classification-false-negative-explanation.png'
    )
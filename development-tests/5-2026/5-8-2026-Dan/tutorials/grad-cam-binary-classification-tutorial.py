"""
Import packages
"""
import imbal
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
from keras import layers, optimizers, metrics

seed = 42
tf.keras.utils.set_random_seed(seed)

"""
Load data
"""
SDO_DATA_PATH = '../../../../tutorials/data/SDOBenchmark' # Ensure data is located at this path

MODEL_SAVE_PATH = 'sdo_binary_classification_model.keras'
LOAD_SAVED_MODEL = True


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


"""
Compile and train/load model
"""
LEARNING_RATE = 5e-5
EPOCHS = 400
BATCH_SIZE = 256

if LOAD_SAVED_MODEL and os.path.exists(MODEL_SAVE_PATH):
    print(f'Loading saved binary classification model from {MODEL_SAVE_PATH}')
    model = keras.models.load_model(
        MODEL_SAVE_PATH,
        custom_objects={'Model': imbal.classification.Model}
    )
else:
    model = build_simple_cnn()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[metrics.F1Score(threshold=0.5)],
    )

    sample_weights = imbal.classification.generate_sample_weights(
        y_train,
        class_weight={0: 0.7, 1: 0.3}
    )

    (x_train, y_train, sw_train), (x_val, y_val, sw_val) = imbal.classification.split(
        x_train,
        y_train,
        sample_weights=sample_weights,
        test_size=0.2
    )

    model.balanced_fit(
        x_train,
        y_train.reshape(-1, 1),
        sample_weight=sw_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        stratify_batches=True, # Ensure all batches have a similar data distribution
        validation_data=(x_val, y_val.reshape(-1, 1), sw_val)
    )

    model.save(MODEL_SAVE_PATH)
    print(f'Saved binary classification model to {MODEL_SAVE_PATH}')

model.evaluate(x_test, y_test.reshape(-1, 1))

"""
Data and results visualization
"""
class_names = ["not X-class", "X-class"]

print('Number of test samples in class 0 (not X-class):', np.sum(y_test == 0))
print('Number of test samples in class 1 (X-class):', np.sum(y_test == 1))

# Predict on test data
test_predictions = model.predict(x_test)
predicted_probabilities = test_predictions.reshape(-1)
predicted_classes = (predicted_probabilities >= 0.5).astype(np.int32)

y_test_flat = y_test.reshape(-1)

# Correct predictions
correct_mask = predicted_classes == y_test_flat

# Incorrect predictions
incorrect_mask = predicted_classes != y_test_flat

correct_indices = np.where(correct_mask)[0]
incorrect_indices = np.where(incorrect_mask)[0]

print("Number correct:", len(correct_indices))
print("Number incorrect:", len(incorrect_indices))

# True positive prediction from the tutorial run
print()
print('Explaining true positive sample')
print('Selected index:', 7)
print('Actual:', int(y_test_flat[7]), class_names[int(y_test_flat[7])])
print('Predicted:', int(predicted_classes[7]), class_names[int(predicted_classes[7])])
print('Prediction probability:', predicted_probabilities[7])

imbal.classification.gradcam_explain_image_sample(
    sample=x_test[7],
    model=model,
    class_names=class_names,
    actual_label=int(y_test_flat[7]),
    label_to_explain=1,
    show=True,
    save_figure=True,
    figure_save_path='grad-cam-classification-binary-true-positive.png'
)

# False positive prediction from the tutorial run
print()
print('Explaining false positive sample')
print('Selected index:', 3)
print('Actual:', int(y_test_flat[3]), class_names[int(y_test_flat[3])])
print('Predicted:', int(predicted_classes[3]), class_names[int(predicted_classes[3])])
print('Prediction probability:', predicted_probabilities[3])

imbal.classification.gradcam_explain_image_sample(
    sample=x_test[3],
    model=model,
    class_names=class_names,
    actual_label=int(y_test_flat[3]),
    label_to_explain=1,
    show=True,
    save_figure=True,
    figure_save_path='grad-cam-classification-binary-false-positive.png'
)

# True negative prediction from the tutorial run
print()
print('Explaining true negative sample')
print('Selected index:', 331)
print('Actual:', int(y_test_flat[331]), class_names[int(y_test_flat[331])])
print('Predicted:', int(predicted_classes[331]), class_names[int(predicted_classes[331])])
print('Prediction probability:', predicted_probabilities[331])

imbal.classification.gradcam_explain_image_sample(
    sample=x_test[331],
    model=model,
    class_names=class_names,
    actual_label=int(y_test_flat[331]),
    label_to_explain=0,
    show=True,
    save_figure=True,
    figure_save_path='grad-cam-classification-binary-true-negative.png'
)

# False negative prediction from the tutorial run
print()
print('Explaining false negative sample')
print('Selected index:', 333)
print('Actual:', int(y_test_flat[333]), class_names[int(y_test_flat[333])])
print('Predicted:', int(predicted_classes[333]), class_names[int(predicted_classes[333])])
print('Prediction probability:', predicted_probabilities[333])

imbal.classification.gradcam_explain_image_sample(
    sample=x_test[333],
    model=model,
    class_names=class_names,
    actual_label=int(y_test_flat[333]),
    label_to_explain=0,
    show=True,
    save_figure=True,
    figure_save_path='grad-cam-classification-binary-false-negative.png'
)

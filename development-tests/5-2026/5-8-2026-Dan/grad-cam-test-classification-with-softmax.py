"""
Import packages
"""
import imbal
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
from keras import layers, optimizers, callbacks, metrics

seed = 42
tf.keras.utils.set_random_seed(
    seed
)

"""
Load data
"""
SDO_DATA_PATH = '../../../tutorials/data/SDOBenchmark' # Ensure data is located at this path

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

def flux_to_sdo_class(y):
    # 0 = everything else / large majority
    # 1 = M-class / larger minority: -5 <= flux <= -4
    # 2 = X-class / small minority: flux > -4
    labels = np.zeros_like(y, dtype=np.int32)
    labels[(y >= -5) & (y <= -4)] = 1
    labels[y > -4] = 2
    return labels

# Load train and test data via function defined above
x_train, y_train = load_sdo_data(os.path.join(SDO_DATA_PATH, 'training'))
x_test, y_test = load_sdo_data(os.path.join(SDO_DATA_PATH, 'test'))
y_train = flux_to_sdo_class(y_train)
y_test = flux_to_sdo_class(y_test)

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
    output_layer = layers.Dense(3, activation='softmax')(x)

    model = imbal.classification.Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    return model

model = build_simple_cnn()

"""
Compile and train model
"""
LEARNING_RATE = 5e-5
EPOCHS = 200
BATCH_SIZE = 256

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

sample_weights = imbal.classification.generate_sample_weights(y_train, class_weight={0: 0.9, 1: 0.1})
(x_train, y_train, sw_train), (x_val, y_val, sw_val) = imbal.classification.split(x_train, y_train, sample_weights=sample_weights, test_size=0.2)

model.balanced_fit(
    x_train,
    y_train.reshape(-1, 1),
    sample_weight=sw_train,
    #class_weight={0: 0.9, 1: 0.1},
    # class_weight=[[0.9, 0.1,], [0.6, 0.4], [0.5, 0.5]], # Uncomment to use varying class weights
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    stratify_batches=True, # Ensure all batches have a similar data distribution
    validation_data=(x_val, y_val.reshape(-1, 1), sw_val)
)

model.evaluate(x_test, y_test.reshape(-1, 1))

"""
Data and results visualization
"""

class_names = ["not M- nor X-class", "M-class", "X-class"]

for class_id, class_name in enumerate(class_names):
    print(f'Number of test samples in class {class_id} ({class_name}):', np.sum(y_test == class_id))

# Predict on test data
test_predictions = model.predict(x_test)
predicted_classes = np.argmax(test_predictions, axis=1)

y_test_flat = y_test.reshape(-1)
predicted_classes = predicted_classes.reshape(-1)

# Correct predictions
correct_mask = predicted_classes == y_test_flat

# Incorrect predictions
incorrect_mask = predicted_classes != y_test_flat

correct_indices = np.where(correct_mask)[0]
incorrect_indices = np.where(incorrect_mask)[0]

print("Number correct:", len(correct_indices))
print("Number incorrect:", len(incorrect_indices))


def corner_intensity_features(img, patch=10):
    """
    Extract corner-based similarity features.

    This does NOT assume corners are black.
    It compares the gray/intensity structure of the four corners.
    """
    img = img.squeeze()

    corners = [
        img[:patch, :patch],        # top-left
        img[:patch, -patch:],       # top-right
        img[-patch:, :patch],       # bottom-left
        img[-patch:, -patch:]       # bottom-right
    ]

    features = []

    for corner in corners:
        features.extend([
            np.mean(corner),        # average gray level
            np.std(corner),         # variation / texture
            np.min(corner),         # darkest value
            np.max(corner)          # brightest value
        ])

    return np.array(features)


def find_most_similar_correct_and_incorrect_pair_for_class(class_id):
    """
    Find one correct and one incorrect prediction for a specific actual class.

    This keeps the professor's requested class instance clear:
    both samples have actual_label == class_id.

    The pair is selected based on similar corner intensity features so that
    the correct and incorrect examples are visually comparable.
    """
    class_correct_indices = np.where(
        (y_test_flat == class_id) &
        (predicted_classes == class_id)
    )[0]

    class_incorrect_indices = np.where(
        (y_test_flat == class_id) &
        (predicted_classes != class_id)
    )[0]

    print(
        f'Class {class_id} ({class_names[class_id]}): '
        f'{len(class_correct_indices)} correct, '
        f'{len(class_incorrect_indices)} incorrect'
    )

    if len(class_correct_indices) == 0 or len(class_incorrect_indices) == 0:
        return None, None

    correct_features = np.array([
        corner_intensity_features(x_test[i], patch=10)
        for i in class_correct_indices
    ])

    incorrect_features = np.array([
        corner_intensity_features(x_test[i], patch=10)
        for i in class_incorrect_indices
    ])

    best_distance = np.inf
    best_pair = None

    for a, correct_i in enumerate(class_correct_indices):
        for b, incorrect_i in enumerate(class_incorrect_indices):

            distance = np.linalg.norm(
                correct_features[a] - incorrect_features[b]
            )

            if distance < best_distance:
                best_distance = distance
                best_pair = (correct_i, incorrect_i)

    return best_pair


def explain_sample(sample_index, class_id, prediction_result):
    """
    Generate and save a Grad-CAM explanation for one sample.
    """
    actual_class = y_test_flat[sample_index]
    predicted_class = predicted_classes[sample_index]
    prediction_confidence = test_predictions[sample_index][predicted_class]

    print()
    print(f'Explaining {prediction_result} prediction for class {class_id} ({class_names[class_id]})')
    print('Selected index:', sample_index)
    print('Actual:', actual_class, class_names[actual_class])
    print('Predicted:', predicted_class, class_names[predicted_class])
    print('Prediction confidence:', prediction_confidence)

    imbal.classification.gradcam_explain_image_sample(
        sample=x_test[sample_index],
        model=model,
        class_names=class_names,
        actual_label=actual_class,
        show=True,
        save_figure=True,
        figure_save_path=f'grad-cam-classification-softmax-class-{class_id}-{prediction_result}.png'
    )

for class_id in range(3):
    correct_idx, incorrect_idx = find_most_similar_correct_and_incorrect_pair_for_class(class_id)

    if correct_idx is None or incorrect_idx is None:
        print()
        print(
            f'Could not find both a correct and incorrect prediction for '
            f'class {class_id} ({class_names[class_id]}).'
        )
        print(
            'You may need a different train/test split, different model checkpoint, '
            'or different training settings to get both cases for this class.'
        )
        continue

    explain_sample(
        sample_index=correct_idx,
        class_id=class_id,
        prediction_result='correct'
    )

    explain_sample(
        sample_index=incorrect_idx,
        class_id=class_id,
        prediction_result='incorrect'
    )
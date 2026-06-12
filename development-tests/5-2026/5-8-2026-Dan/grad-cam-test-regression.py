"""
Import packages
"""
import imbal
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import layers, optimizers, callbacks, metrics

seed = 42
tf.keras.utils.set_random_seed(
    seed
)

"""
Load data
"""
SDO_DATA_PATH = '../../../tutorials/data/SDOBenchmark' # Ensure data is located at this path
KDE_BIN_COUNT = 32

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
    output_layer = layers.Dense(1)(x)

    model = imbal.regression.Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    return model

MODEL_PATH = "sdo_regression_model.keras"

"""
Build/load model
"""
if os.path.exists(MODEL_PATH):
    print(f"Loading saved model from {MODEL_PATH}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "Model": imbal.regression.Model
        },
    )

else:
    print("No saved model found. Training a new model.")

    model = build_simple_cnn()

    """
    Calculate data density distribution, and extract sample densities
    """

    data_kde_bandwidth = imbal.regression.fit_kde(y_train, bin_count=KDE_BIN_COUNT)
    sample_densities = imbal.regression.get_sample_densities(y_train, data_kde_bandwidth)

    """
    Compile and train model
    """
    LEARNING_RATE = 5e-5
    EPOCHS = 400
    BATCH_SIZE = 256

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    model.balanced_fit(
        x_train,
        y_train,
        sample_density=sample_densities,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        stratify_batches=True
    )

    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

model.evaluate(x_test, y_test.reshape(-1))

"""
Probability Density Distribution and Results Visualization
"""
test_rare_mask = y_test > -4
test_frequent_mask = ~test_rare_mask
print('Number of test samples with log10 flux < -4:', np.sum(test_frequent_mask.astype(np.int32)))
print('Number of test samples with log10 flux >= -4:', np.sum(test_rare_mask.astype(np.int32)))

# Predict on test data
test_predictions = model.predict(x_test)

test_predictions_rare = test_predictions[test_rare_mask] # Mask rare test data
test_labels_rare = y_test[test_rare_mask] # Mask predictions on rare test data
test_predictions_frequent = test_predictions[test_frequent_mask] # Mask frequent test data
test_labels_frequent = y_test[test_frequent_mask] # Mask predictions on frequent test data

# Calculate metrics
overall_test_mae = np.mean(np.abs(test_predictions - y_test))
frequent_test_mae = np.mean(np.abs(test_predictions_frequent - test_labels_frequent))
rare_test_mae = np.mean(np.abs(test_predictions_rare - test_labels_rare))

print(
    f'MAE for log10 flux < -4: {frequent_test_mae:.3f}\n'
    f'MAE for log10 flux >= -4: {rare_test_mae:.3f}'
)

data_kde_bandwidth = imbal.regression.fit_kde(y_train, bin_count=KDE_BIN_COUNT)

imbal.regression.plot_kde_1d(
    y_train,
    data_kde_bandwidth,
    bin_count=KDE_BIN_COUNT,
    show_bin_count=False,
    save_figure='sample-sdo-balanced-fit-data-distribution.png'
)

imbal.regression.plot_true_vs_predictions(
    y_test,
    test_predictions,
    save_figure='sample-sdo-balanced-fit-label-vs-prediction-plot.png'
)

pred = test_predictions.reshape(-1)
true = y_test.reshape(-1)
error = np.abs(pred - true)


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


# Good example candidates: top-right and close to diagonal
good_candidates = np.where(
    (true > -5.0) &
    (pred > -5.0) &
    (error < 0.1)
)[0]

# Bad example candidates: around x = -5, but far below diagonal
bad_candidates = np.where(
    (true > -5.5) &
    (true < -4.5) &
    (pred < true - 2.0)
)[0]

print("Number of good regression candidates:", len(good_candidates))
print("Number of bad regression candidates:", len(bad_candidates))

if len(good_candidates) > 0 and len(bad_candidates) > 0:
    good_features = np.array([
        corner_intensity_features(x_test[i], patch=10)
        for i in good_candidates
    ])

    bad_features = np.array([
        corner_intensity_features(x_test[i], patch=10)
        for i in bad_candidates
    ])

    best_distance = np.inf
    best_pair = None

    for a, good_i in enumerate(good_candidates):
        for b, bad_i in enumerate(bad_candidates):

            distance = np.linalg.norm(
                good_features[a] - bad_features[b]
            )

            if distance < best_distance:
                best_distance = distance
                best_pair = (good_i, bad_i)

    good_idx, bad_idx = best_pair

    print("Selected similar good regression index:", good_idx)
    print("Good true:", true[good_idx])
    print("Good pred:", pred[good_idx])
    print("Good error:", error[good_idx])

    print("Selected similar bad regression index:", bad_idx)
    print("Bad true:", true[bad_idx])
    print("Bad pred:", pred[bad_idx])
    print("Bad error:", error[bad_idx])

    print("Corner feature distance:", best_distance)

    imbal.regression.gradcam_explain_image_sample(
        sample=x_test[good_idx],
        model=model,
        actual_value=y_test[good_idx],
        show=True,
        save_figure=True,
        figure_save_path='images/grad-cam-regression-good-example-similar-corners.png',
        positive_importance_threshold=0.05,
        negative_importance_threshold=0.5
    )

    imbal.regression.gradcam_explain_image_sample(
        sample=x_test[bad_idx],
        model=model,
        actual_value=y_test[bad_idx],
        show=True,
        save_figure=True,
        figure_save_path='images/grad-cam-regression-bad-example-similar-corners.png',
        positive_importance_threshold=0.05,
        negative_importance_threshold=0.5
    )

else:
    print("Could not find both good and bad regression candidates.")
    print("Try loosening the candidate filters if this happens.")
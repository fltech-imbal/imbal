"""
Run Grad-CAM threshold tests using a saved model.

First run grad-cam-test-regression-save-model.py once to create:
    sdo_regression_model.keras

Then run this file whenever you want to test different positive/negative
importance thresholds without retraining the model.
"""
import imbal
import os
import numpy as np
from PIL import Image
import tensorflow as tf

seed = 42
tf.keras.utils.set_random_seed(seed)

SDO_DATA_PATH = '../../../tutorials/data/SDOBenchmark' # Ensure data is located at this path
MODEL_SAVE_PATH = 'sdo_regression_model.keras'


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


# Only test data is needed for Grad-CAM threshold experiments.
x_test, y_test = load_sdo_data(os.path.join(SDO_DATA_PATH, 'test'))

# Load the already-trained model. compile=False is fine because this file only
# predicts and generates Grad-CAM explanations; it does not train.
model = tf.keras.models.load_model(
    MODEL_SAVE_PATH,
    custom_objects={'Model': imbal.regression.Model},
    compile=False,
)
print(f'Loaded saved model from {MODEL_SAVE_PATH}')

# Predict on test data, then select the same good/bad examples as before.
test_predictions = model.predict(x_test)

pred = test_predictions.reshape(-1)
true = y_test.reshape(-1)
error = np.abs(pred - true)

good_candidates = np.where(
    (true > -5.0) &
    (pred > -5.0) &
    (error < 0.25)
)[0]

if len(good_candidates) == 0:
    raise ValueError('No good candidates found. Adjust the good_candidates filters.')

good_idx = good_candidates[np.argmin(error[good_candidates])]

bad_candidates = np.where(
    (true > -5.5) &
    (true < -4.5) &
    (pred < true - 1.0)
)[0]

if len(bad_candidates) == 0:
    raise ValueError('No bad candidates found. Adjust the bad_candidates filters.')

bad_idx = bad_candidates[np.argmax(error[bad_candidates])]

print('Good index:', good_idx, 'true:', true[good_idx], 'pred:', pred[good_idx], 'error:', error[good_idx])
print('Bad index:', bad_idx, 'true:', true[bad_idx], 'pred:', pred[bad_idx], 'error:', error[bad_idx])

# Edit this list to quickly compare different threshold settings.
threshold_tests = [
    {'positive': 0.01, 'negative': 0.01},
    #{'positive': 0.1, 'negative': 0.5},
    #{'positive': 0.25, 'negative': 0.5},
    #{'positive': 0.5, 'negative': 0.5},
]

"""
Create output directory for Grad-CAM images
"""
OUTPUT_DIR = "gradcam-threshold-test-outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cmap = "jet"

for thresholds in threshold_tests:
    positive_threshold = thresholds['positive']
    negative_threshold = thresholds['negative']
    suffix = f'pos-{positive_threshold}_neg-{negative_threshold}'.replace('.', 'p')

    imbal.regression.gradcam_explain_image_sample(
        sample=x_test[good_idx],
        model=model,
        actual_value=y_test[good_idx],
        show=False,
        save_figure=True,
        figure_save_path=os.path.join(
            OUTPUT_DIR,
            f'grad-cam-coloring-{cmap}-good-{suffix}.png'
        ),
        positive_importance_threshold=positive_threshold,
        negative_importance_threshold=negative_threshold,
        cmap=cmap,
    )

    imbal.regression.gradcam_explain_image_sample(
        sample=x_test[bad_idx],
        model=model,
        actual_value=y_test[bad_idx],
        show=False,
        save_figure=True,
        figure_save_path=os.path.join(
            OUTPUT_DIR,
            f'grad-cam-coloring-{cmap}-bad-{suffix}.png'
        ),
        positive_importance_threshold=positive_threshold,
        negative_importance_threshold=negative_threshold,
        cmap=cmap,
    )

    print(f'Saved Grad-CAM figures for positive={positive_threshold}, negative={negative_threshold}')

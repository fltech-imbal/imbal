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

TARGET_OFFSET = 10.0  # Added to all regression targets to create positive-valued samples

def load_sdo_data(data_path):
    # Load labels (log peak flux)
    with open(os.path.join(data_path, 'log_peak_flux.txt'), 'r') as file:
        contents = file.read().strip()
        loaded_data_fluxes = np.array([float(x) for x in contents.split('\n')])

    # Shift targets upward so Grad-CAM explanations can include positive contributions
    loaded_data_fluxes = loaded_data_fluxes + TARGET_OFFSET

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

print(f'Applied target offset of +{TARGET_OFFSET} to all regression labels')

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

model = build_simple_cnn()

"""
Calculate data density distribution, and extract sample densities
"""
KDE_BIN_COUNT=32

# Determine KDE fit for data, then extract sample densities
data_kde_bandwidth = imbal.regression.fit_kde(y_train, bin_count=KDE_BIN_COUNT)
sample_densities = imbal.regression.get_sample_densities(y_train, data_kde_bandwidth)

# The below line can be uncommented to test multiple alpha values for reciprocal importance
# If this is uncommented, be sure to also uncomment 'sample_weight=sample_weight_candidates' in the following section
# sample_weight_candidates = imbal.regression.reciprocal_importance(sample_densities, alpha=[0.2, 0.5, 1.0])

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
    # sample_weight=sample_weight_candidates, # Uncomment to use varying alphas for reciprocal importance (see above section)
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    stratify_batches=True # Ensure all batches have a similar data distribution
)

model.save("sdo_regression_model.keras")

model.evaluate(x_test, y_test.reshape(-1))

"""
Probability Density Distribution and Results Visualization
"""
test_rare_mask = y_test > 6
# Equivalent to original threshold of -4 after applying +10 offset

test_frequent_mask = ~test_rare_mask
print('Number of test samples with shifted log10 flux < 6:', np.sum(test_frequent_mask.astype(np.int32)))
print('Number of test samples with shifted log10 flux >= 6:', np.sum(test_rare_mask.astype(np.int32)))

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
    f'MAE for shifted log10 flux < 6: {frequent_test_mae:.3f}\n'
    f'MAE for shifted log10 flux >= 6: {rare_test_mae:.3f}'
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

# Good example: top-right and close to diagonal
good_candidates = np.where(
    (true > 5.0) &
    (pred > 5.0) &
    (error < 0.1)
)[0]

good_idx = good_candidates[np.argmin(error[good_candidates])]

# Bad example: around x = 5, but far below diagonal
bad_candidates = np.where(
    (true > 4.5) &
    (true < 5.5) &
    (pred < true - 1.25)
)[0]

bad_idx = bad_candidates[np.argmax(error[bad_candidates])]

print("Good index:", good_idx, "true:", true[good_idx], "pred:", pred[good_idx], "error:", error[good_idx])
print("Bad index:", bad_idx, "true:", true[bad_idx], "pred:", pred[bad_idx], "error:", error[bad_idx])

imbal.regression.gradcam_explain_image_sample(
    sample=x_test[good_idx],
    model=model,
    actual_value=y_test[good_idx],
    show=True,
    save_figure=True,
    figure_save_path='images/grad-cam-regression-explanation.png',
    positive_importance_threshold=0.05,
    negative_importance_threshold=0.5
)

imbal.regression.gradcam_explain_image_sample(
    sample=x_test[bad_idx],
    model=model,
    actual_value=y_test[bad_idx],
    show=True,
    save_figure=True,
    figure_save_path='images/grad-cam-regression-bad-example.png',
    positive_importance_threshold=0.05,
    negative_importance_threshold=0.5
)

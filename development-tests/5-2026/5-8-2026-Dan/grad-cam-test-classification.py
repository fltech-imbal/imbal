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

MODEL_PATH = "sdo_binary_classification_model.keras"

"""
Build/load model
"""
if os.path.exists(MODEL_PATH):
    print(f"Loading saved model from {MODEL_PATH}")

    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={'Model': imbal.classification.Model}
    )
else:
    print("No saved model found. Training a new model.")

    model = build_simple_cnn()

    """
    Compile and train model
    """
    LEARNING_RATE = 5e-5
    EPOCHS = 400
    BATCH_SIZE = 256

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[metrics.F1Score(threshold=0.5)],
    )

    sample_weights = imbal.classification.generate_sample_weights(
        y_train,
        class_weight={0: 0.7, 1: 0.3}
    )

    (x_train, y_train, sw_train), (x_val, y_val, sw_val) = (
        imbal.classification.split(
            x_train,
            y_train,
            sample_weights=sample_weights,
            test_size=0.2
        )
    )

    model.balanced_fit(
        x_train,
        y_train.reshape(-1, 1),
        sample_weight=sw_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        stratify_batches=True,
        validation_data=(x_val, y_val.reshape(-1, 1), sw_val)
    )

    # (x_train, y_train), (x_val, y_val) = (
    #     imbal.classification.split(
    #         x_train,
    #         y_train,
    #         test_size=0.2
    #     )
    # )
    #
    # model.balanced_fit(
    #     x_train,
    #     y_train.reshape(-1, 1),
    #     epochs=EPOCHS,
    #     batch_size=BATCH_SIZE,
    #     stratify_batches=True,
    #     validation_data=(x_val, y_val.reshape(-1, 1)),
    #     class_weight=[[0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2]]
    # )

    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

model.evaluate(x_test, y_test.reshape(-1, 1))

"""
Data and results visualization
"""

test_rare_mask = y_test == 1
test_frequent_mask = ~test_rare_mask
print('Number of test samples with log10 flux < -4:', np.sum(test_frequent_mask))
print('Number of test samples with log10 flux >= -4:', np.sum(test_rare_mask))

# Predict on test data
test_predictions = model.predict(x_test)
test_predictions = test_predictions.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Calculate metrics
hss = imbal.metrics.HeidkeSkillScore(threshold=0.5)
hss.update_state(y_test, test_predictions)

f1 = keras.metrics.F1Score(threshold=0.5)
f1.update_state(y_test, test_predictions)

print(
    f'Heikde Skill Score: {hss.result()[0]:.4f}\n'
    f'F1 Score: {f1.result()[0]:.4f}\n'
)

imbal.classification.plot_confusion_matrix(
    y_test,
    test_predictions,
    save_figure='classification-explanation-confusion-matrix.png'
)

rounded_predictions = np.round(test_predictions).astype(np.int32).reshape(-1)
y_test_flat = y_test.reshape(-1)

true_positive_mask = (y_test_flat == 1) & (rounded_predictions == 1)
false_positive_mask = (y_test_flat == 0) & (rounded_predictions == 1)
true_negative_mask = (y_test_flat == 0) & (rounded_predictions == 0)
false_negative_mask = (y_test_flat == 1) & (rounded_predictions == 0)

tp_indices = np.where(true_positive_mask)[0]
fp_indices = np.where(false_positive_mask)[0]
tn_indices = np.where(true_negative_mask)[0]
fn_indices = np.where(false_negative_mask)[0]

print("Number of true positives:", len(tp_indices))
print("Number of false positives:", len(fp_indices))
print("Number of true negatives:", len(tn_indices))
print("Number of false negatives:", len(fn_indices))


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


if len(tp_indices) > 0 and len(fp_indices) > 0 and len(tn_indices) > 0 and len(fn_indices) > 0:
    tp_features = np.array([
        corner_intensity_features(x_test[i], patch=10)
        for i in tp_indices
    ])

    fp_features = np.array([
        corner_intensity_features(x_test[i], patch=10)
        for i in fp_indices
    ])

    tn_features = np.array([
        corner_intensity_features(x_test[i], patch=10)
        for i in tn_indices
    ])

    fn_features = np.array([
        corner_intensity_features(x_test[i], patch=10)
        for i in fn_indices
    ])

    best_distance = np.inf
    best_group = None

    for a, tp_i in enumerate(tp_indices):
        for b, fp_i in enumerate(fp_indices):
            for c, tn_i in enumerate(tn_indices):
                for d, fn_i in enumerate(fn_indices):
                    distance = (
                            np.linalg.norm(tp_features[a] - fp_features[b]) +
                            np.linalg.norm(tp_features[a] - tn_features[c]) +
                            np.linalg.norm(tp_features[a] - fn_features[d]) +
                            np.linalg.norm(fp_features[b] - tn_features[c]) +
                            np.linalg.norm(fp_features[b] - fn_features[d]) +
                            np.linalg.norm(tn_features[c] - fn_features[d])
                    )

                    if distance < best_distance:
                        best_distance = distance
                        best_group = (tp_i, fp_i, tn_i, fn_i)

    tp_idx, fp_idx, tn_idx, fn_idx = best_group

    print("Selected similar true positive index:", tp_idx)
    print("Selected similar false positive index:", fp_idx)
    print("Selected similar true negative index:", tn_idx)
    print("Selected similar false negative index:", fn_idx)
    print("Total corner feature distance:", best_distance)

    print("TP actual label:", y_test_flat[tp_idx])
    print("TP predicted probability:", test_predictions.reshape(-1)[tp_idx])

    print("FP actual label:", y_test_flat[fp_idx])
    print("FP predicted probability:", test_predictions.reshape(-1)[fp_idx])

    print("TN actual label:", y_test_flat[tn_idx])
    print("TN predicted probability:", test_predictions.reshape(-1)[tn_idx])

    print("FN actual label:", y_test_flat[fn_idx])
    print("FN predicted probability:", test_predictions.reshape(-1)[fn_idx])

    imbal.classification.gradcam_explain_image_sample(
        sample=x_test[tp_idx],
        model=model,
        class_names=["not X-class", "X-class"],
        actual_label=1,
        show=True,
        save_figure=True,
        figure_save_path='images/grad-cam-classification-true-positive-similar-corners.png'
    )

    imbal.classification.gradcam_explain_image_sample(
        sample=x_test[fp_idx],
        model=model,
        class_names=["not X-class", "X-class"],
        actual_label=0,
        show=True,
        save_figure=True,
        figure_save_path='images/grad-cam-classification-false-positive-similar-corners.png'
    )

    imbal.classification.gradcam_explain_image_sample(
        sample=x_test[tn_idx],
        model=model,
        class_names=["not X-class", "X-class"],
        actual_label=0,
        show=True,
        save_figure=True,
        figure_save_path='images/grad-cam-classification-true-negative-similar-corners.png'
    )

    imbal.classification.gradcam_explain_image_sample(
        sample=x_test[fn_idx],
        model=model,
        class_names=["not X-class", "X-class"],
        actual_label=1,
        show=True,
        save_figure=True,
        figure_save_path='images/grad-cam-classification-false-negative-similar-corners.png'
    )

else:
    print("Could not find true positives, false positives, true negatives, and false negatives.")

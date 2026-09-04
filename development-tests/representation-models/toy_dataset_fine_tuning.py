"""
Import packages
"""
import imbal
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import pandas as pd
from tools import FitType
"""
Set script parameters
"""

LEARNING_RATE = 5e-5
FIT = FitType.BALANCED
VALIDATION_DATA = True
AE = False
AE_THIRD_TO_LAST = False
WEIGHT_CANDIDATES = False
SINGLE_WEIGHT_ALPHA = 1

REPRESENTATION_LAYER_INDEX = -2
EARLY_STOPPING_PATIENCE = 100
EPOCHS = 10000

DATA_PATH = "cleaned-dtw-SEP-EC-data"
DATA_PREFIX = 'toy_dataset'
OUTPUT_PATH = "results"
OUTPUT_POSTFIX = '_fine_tuned_balanced_2'
USE_DELTA = False

# Will be mostly left unchanged
STRATIFY = True
BATCH_SIZE = 2048
KDE_BIN_COUNT=64
SEED = 42

"""
Load data
"""

x_train = tf.clip_by_value(tf.random.normal((1000, 2)), clip_value_min=-3, clip_value_max=3)
x_train = x_train.numpy()
y_train = tf.reshape(tf.linalg.norm(x_train, axis=1), (-1, 1))
y_train = y_train.numpy()
x_val = tf.clip_by_value(tf.random.normal((250, 2)), clip_value_min=-3, clip_value_max=3)
x_val = x_val.numpy()
y_val = tf.reshape(tf.linalg.norm(x_val, axis=1), (-1, 1))
y_val = y_val.numpy()
x_test = tf.clip_by_value(tf.random.normal((250, 2)), clip_value_min=-3, clip_value_max=3)
x_test = x_test.numpy()
y_test = tf.reshape(tf.linalg.norm(x_test, axis=1), (-1, 1))
y_test = y_test.numpy()

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

print(y_train[y_train > np.log(10)].shape)
print(y_train[y_train <= np.log(10)].shape)
print(y_test[y_test > np.log(10)].shape)
print(y_test[y_test <= np.log(10)].shape)

print()

from imbal.util.backend.constants import ModelType
temp = imbal.util.backend.DatasetWithBatching(
    x_train,
    y_train,
    mode=ModelType.REGRESSION
)

print('test')
print(temp[0][1])

"""
Build model
"""

if FIT == FitType.REGULAR:
    WEIGHT_CANDIDATES = False

# tf.keras.utils.set_random_seed(
#     SEED
# )

# tf.config.run_functions_eagerly(True)

LAYER_DIMS = [64, 64, 64, 32, 32, 2]

inputs = keras.Input(shape=(x_train.shape[1],))

x = inputs
for index, num_units in enumerate(LAYER_DIMS):
    # if index == len(LAYER_DIMS) - 1 and not AE_THIRD_TO_LAST:
    #     x = layers.Flatten()(x)
    # if index == len(LAYER_DIMS) - 2 and AE_THIRD_TO_LAST:
    #     x = layers.Flatten()(x)
    x = layers.Dense(num_units, activation='relu')(x)

outputs = layers.Dense(1)(x)

model = imbal.regression.Model(inputs=inputs, outputs=outputs, name="SEP_EC")

def safe_norm(x, axis):
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis) + 1e-12)

def cauchy_schwartz(labels, representations, weight=None):
    # print(labels)
    distance_to_next_label = tf.abs(labels[1:] - labels[:-1])
    distance_to_first_label = tf.abs(labels[1:] - labels[0])

    distance_to_next_representation = safe_norm(representations[1:] - representations[:-1], axis=1)
    distance_to_first_representation = safe_norm(representations[1:] - representations[0], axis=1)

    combined_label_distances = tf.concat([distance_to_next_label, distance_to_first_label], axis=0)
    combined_label_distances = tf.squeeze(combined_label_distances)
    combined_representation_distances = tf.concat(
        [distance_to_next_representation, distance_to_first_representation],
        axis=0)
    a = combined_label_distances
    b = combined_representation_distances

    n = tf.cast(tf.size(a), tf.float32)

    return (
                   tf.reduce_sum(a * a) * tf.reduce_sum(b * b)
                   - tf.square(tf.reduce_sum(a * b))
           ) / tf.square(n)

def stub_function(y_true, y_pred, weights=None):
    return 0

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    weighted_metrics=['mae'],
    generate_decoder_branch=AE,
    representation_layer_index=-3 if AE_THIRD_TO_LAST else -2,
    representation_loss=cauchy_schwartz
)

# if FIT == FitType.DECOUPLED:
#     model.override_second_stage_fit_parameters(
#         callbacks=[keras.callbacks.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, min_delta=1e-5)] if VALIDATION_DATA else None
#     )

"""
Generate sample densities
"""


fit_function = model.fit
if FIT == FitType.BALANCED:
    fit_function = model.balanced_fit
if FIT == FitType.DECOUPLED:
    fit_function = model.rRT_fit

if VALIDATION_DATA:
    kde_bandwidth = imbal.regression.fit_kde(
        y_train,
        bin_count=KDE_BIN_COUNT
    )

    sample_densities = imbal.regression.get_sample_densities(
        y_train,
        kde_bandwidth,
    )
    sample_weights = imbal.regression.reciprocal_importance(sample_densities, alpha=[0.1*(i+1) for i in range(10)] if WEIGHT_CANDIDATES else SINGLE_WEIGHT_ALPHA)
    val_densities = imbal.regression.get_sample_densities(
        y_val,
        kde_bandwidth,
        distribution=y_train
    )
    w_val = imbal.regression.reciprocal_importance(val_densities, alpha=[0.1*(i+1) for i in range(10)] if WEIGHT_CANDIDATES else SINGLE_WEIGHT_ALPHA)
    val_data = (x_val, y_val, w_val)
else:
    x_train = np.concatenate((x_train, x_val))
    y_train = np.concatenate((y_train, y_val))

    kde_bandwidth = imbal.regression.fit_kde(
        y_train,
        bin_count=KDE_BIN_COUNT
    )

    sample_densities = imbal.regression.get_sample_densities(
        y_train,
        kde_bandwidth,
    )
    sample_weights = imbal.regression.reciprocal_importance(sample_densities)

    val_data = None

# imbal.regression.plot_kde_1d(
#     y_train,
#     kde_bandwidth,
#     bin_count=KDE_BIN_COUNT
# )

training_history = fit_function(
    x_train,
    y_train,
    sample_weight=sample_weights,
    validation_data=val_data,
    validation_split=None,
    epochs=(EPOCHS, EPOCHS) if FIT == FitType.DECOUPLED else EPOCHS,
    batch_size=BATCH_SIZE,
    candidate_evaluation_sample_weight=(val_data[2][-1] if VALIDATION_DATA else sample_weights[-1]) if WEIGHT_CANDIDATES else None,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss' if VALIDATION_DATA else 'loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, min_delta=1e-5 if VALIDATION_DATA else 1e-3)]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    weighted_metrics=['mae'],
    generate_decoder_branch=AE,
    representation_layer_index=-3 if AE_THIRD_TO_LAST else -2,
    # representation_loss=cauchy_schwartz
)

training_history = fit_function(
    x_train,
    y_train,
    sample_weight=sample_weights,
    validation_data=val_data,
    validation_split=None,
    epochs=(EPOCHS, EPOCHS) if FIT == FitType.DECOUPLED else EPOCHS,
    batch_size=BATCH_SIZE,
    candidate_evaluation_sample_weight=(val_data[2][-1] if VALIDATION_DATA else sample_weights[-1]) if WEIGHT_CANDIDATES else None,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss' if VALIDATION_DATA else 'loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, min_delta=1e-5 if VALIDATION_DATA else 1e-3)]
)

if FIT == FitType.DECOUPLED:
    stage_one_len = len(training_history[0].history['loss'])
    stage_two_len = len(training_history[1].history['loss'])
else:
    stage_one_len = len(training_history.history['loss'])
    stage_two_len = None

predictions = model.predict(x_test)
predictions = predictions.reshape(-1)
y_test = y_test.reshape(-1)

common_sample_mask = (y_test > -0.5) & (y_test < 0.5) if USE_DELTA else (y_test < np.log(10))
common_predictions = predictions[common_sample_mask]
rare_predictions = predictions[~common_sample_mask]
common_labels = y_test[common_sample_mask]
rare_labels = y_test[~common_sample_mask]

mae = np.mean(np.abs(predictions - y_test))
common_mae = np.mean(np.abs(common_predictions - common_labels))
rare_mae = np.mean(np.abs(rare_predictions - rare_labels))

model.save(f"models/{DATA_PREFIX}_{FIT.name.lower()}_{'w' if VALIDATION_DATA or AE else ''}{'_validation' if VALIDATION_DATA else ''}{'_ae' if AE else ''}{'_third_last' if AE_THIRD_TO_LAST and AE else ''}{OUTPUT_POSTFIX}.keras")

if FIT != FitType.REGULAR and VALIDATION_DATA and WEIGHT_CANDIDATES:
    print([0.1*(i+1) for i in range(10)][model.best_weight_index])

print(len(y_train[y_train < np.log(10)]), len(y_train[y_train >= np.log(10)]))
print(len(common_predictions), len(rare_predictions))
print(stage_one_len, stage_two_len, common_mae, rare_mae, (mae + rare_mae)/2, model._reconstruction_lambda)

# print(np.count_nonzero(common_sample_mask), np.count_nonzero(~common_sample_mask))

# imbal.regression.plot_true_vs_predictions(
#     y_test,
#     predictions,
#     title=f'SEP-E - Common MAE: {common_mae:.4f}, Rare MAE: {rare_mae:.4f}, AORE: {(mae + rare_mae)/2:.4f}{f", Alpha: {[0.1*(i+1) for i in range(10)][model.best_weight_index]:.1f}" if WEIGHT_CANDIDATES else ""}',
#     save_figure=f"{OUTPUT_PATH}/{DATA_PREFIX}_{FIT.name.lower()}_{'w' if VALIDATION_DATA or AE else ''}{'_validation' if VALIDATION_DATA else ''}{'_ae' if AE else ''}{'_third_last' if AE_THIRD_TO_LAST and AE else ''}{OUTPUT_POSTFIX}.png"
# )

imbal.regression.tsne_visualization(
    model,
    x_test,
    y_test,
    perplexity=30,
    save_figure=f'toy_dataset_tsne{OUTPUT_POSTFIX}.png'
)
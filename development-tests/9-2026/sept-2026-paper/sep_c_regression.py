"""
Import packages
"""
import imbal
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import pandas as pd
from tools import load_sep_c_data
from tools.loss_functions import *

# tf.config.run_functions_eagerly(True)

"""
Set script parameters
"""

LEARNING_RATE = 1e-3
VALIDATION_DATA = True
AE = False
AE_THIRD_TO_LAST = False
# WEIGHT_CANDIDATES = None
WEIGHT_CANDIDATES = [0.1, 0.3, 0.5, 0.7]
SINGLE_WEIGHT_ALPHA = 1
FIT_MODE = 'joint'
BALANCED_FIRST_STAGE = False
UNIT_REPRESENTATIONS = False
NONLINEAR_REGRESSOR = False
DECORRELATION = False

REPRESENTATION_LAYER_INDEX = -2
EARLY_STOPPING_PATIENCE = 100
EPOCHS = 10000

DATA_PATH = "cleaned-SEP-C-data"
DATA_PREFIX = 'sep_c_w_noise'
OUTPUT_PATH = "results-final-c"
MODEL_OUTPUT_PATH = "models-final-c"
OUTPUT_POSTFIX = '_cosine_joint_weighted_5'
USE_DELTA = False

# Will be mostly left unchanged
STRATIFY = True
BATCH_SIZE = 2048
KDE_BIN_COUNT=64
SEED = 42

"""
Load data
"""

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_sep_c_data(
    f"{DATA_PATH}/{DATA_PREFIX}",
)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

print(y_train[y_train > np.log(10)].shape)
print(y_train[y_train <= np.log(10)].shape)
print(y_test[y_test > np.log(10)].shape)
print(y_test[y_test <= np.log(10)].shape)

y_test = y_test.reshape(-1)
y_test_sort_indices = np.argsort(y_test)
y_test = y_test[y_test_sort_indices]
x_test = x_test[y_test_sort_indices]

# temp = imbal.util.backend.DatasetWithBatching(
#     x_train,
#     y_train,
#     batch_size=64,
#     shuffle=True,
#     mode=imbal.util.backend.constants.ModelType.REGRESSION
# )
#
# print(temp[0])

train_label_min = np.min(y_train)
train_label_max = np.max(y_train)
RATIO_LOSS_LAMBDA = 0
REPRESENTATION_LOSS = weighted_cosine_similarity_w_ratio_loss(train_label_min, train_label_max, RATIO_LOSS_LAMBDA, unit=UNIT_REPRESENTATIONS, decorr=DECORRELATION)

"""
Build model
"""

# tf.keras.utils.set_random_seed(
#     SEED
# )

LAYER_DIMS = [18, 9, 6]

inputs = keras.Input(shape=(x_train.shape[1],))

x = inputs
for index, num_units in enumerate(LAYER_DIMS):
    x = layers.Dense(num_units, activation='relu')(x)
    if index == len(LAYER_DIMS) - 1 and UNIT_REPRESENTATIONS:
        x = layers.UnitNormalization()(x)

if NONLINEAR_REGRESSOR:
    REPRESENTATION_LAYER_INDEX = -6
    for i in range(4):
        x = layers.Dense(6, activation='relu')(x)

outputs = layers.Dense(1)(x)

model = imbal.regression.Model(inputs=inputs, outputs=outputs, name="SEP_EC")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    weighted_metrics=['mae'],
    generate_decoder_branch=AE,
    representation_layer_index=REPRESENTATION_LAYER_INDEX,
    representation_loss=REPRESENTATION_LOSS,
)

# if FIT == FitType.DECOUPLED:
#     model.override_second_stage_fit_parameters(
#         callbacks=[keras.callbacks.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, min_delta=1e-5)] if VALIDATION_DATA else None
#     )

"""
Generate sample densities
"""



if VALIDATION_DATA:
    kde_bandwidth = imbal.regression.fit_kde(
        y_train,
        bin_count=KDE_BIN_COUNT
    )

    sample_densities = imbal.regression.get_sample_densities(
        y_train,
        kde_bandwidth,
    )
    sample_weights = imbal.regression.reciprocal_importance(sample_densities, alpha=WEIGHT_CANDIDATES if WEIGHT_CANDIDATES is not None else SINGLE_WEIGHT_ALPHA)
    val_densities = imbal.regression.get_sample_densities(
        y_val,
        kde_bandwidth,
        distribution=y_train
    )
    w_val = imbal.regression.reciprocal_importance(val_densities, alpha=WEIGHT_CANDIDATES if WEIGHT_CANDIDATES is not None else SINGLE_WEIGHT_ALPHA)
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

training_history = None
best_weight_index = None

if FIT_MODE == 'tune':
    if BALANCED_FIRST_STAGE:
        training_history = model.balanced_fit(
            x_train,
            y_train,
            sample_weight=sample_weights,
            validation_data=val_data,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=False,
            candidate_evaluation_sample_weight=(
                val_data[2][2] if VALIDATION_DATA else sample_weights[-1]) if WEIGHT_CANDIDATES is not None else None,
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss' if VALIDATION_DATA else 'loss',
                                                     patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True,
                                                     min_delta=1e-5 if VALIDATION_DATA else 1e-3)]
        )
        best_weight_index = model.best_weight_index
    else:
        training_history = model.fit(
            x_train,
            y_train,
            shuffle=False,
            validation_data=(val_data[0], val_data[1]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss' if VALIDATION_DATA else 'loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, min_delta=1e-5 if VALIDATION_DATA else 1e-3)]
        )


elif FIT_MODE == 'joint':
    training_history = model.balanced_fit(
        x_train,
        y_train,
        sample_weight=sample_weights,
        validation_data=val_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        candidate_evaluation_sample_weight=(
            val_data[2][2] if VALIDATION_DATA else sample_weights[-1]) if WEIGHT_CANDIDATES is not None else None,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss' if VALIDATION_DATA else 'loss',
                                                 patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True,
                                                 min_delta=1e-5 if VALIDATION_DATA else 1e-3)]
    )
    best_weight_index = model.best_weight_index

stage_one_len = len(training_history.history['loss'])
stage_two_len = None

second_stage_best_weight_index = None
if FIT_MODE == 'tune':
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        weighted_metrics=['mae'],
        generate_decoder_branch=AE,
        representation_layer_index=REPRESENTATION_LAYER_INDEX
    )

    training_history = model.balanced_fit(
        x_train,
        y_train,
        sample_weight=sample_weights,
        validation_data=val_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        candidate_evaluation_sample_weight=(
            val_data[2][2] if VALIDATION_DATA else sample_weights[-1]) if WEIGHT_CANDIDATES is not None else None,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss' if VALIDATION_DATA else 'loss',
                                                 patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True,
                                                 min_delta=1e-5 if VALIDATION_DATA else 1e-3)]
    )

    stage_two_len = len(training_history.history['loss'])
    second_stage_best_weight_index = model.best_weight_index

predictions = model.predict(x_test)
predictions = predictions.reshape(-1)
y_test = y_test.reshape(-1)

common_sample_mask =(y_test < np.log(10))
common_predictions = predictions[common_sample_mask]
rare_predictions = predictions[~common_sample_mask]
common_labels = y_test[common_sample_mask]
rare_labels = y_test[~common_sample_mask]

mae = np.mean(np.abs(predictions - y_test))
common_mae = np.mean(np.abs(common_predictions - common_labels))
rare_mae = np.mean(np.abs(rare_predictions - rare_labels))

model.save(f"{MODEL_OUTPUT_PATH}/{DATA_PREFIX}_{'w' if VALIDATION_DATA or AE else ''}{'_validation' if VALIDATION_DATA else ''}{'_ae' if AE else ''}{'_third_last' if AE_THIRD_TO_LAST and AE else ''}{OUTPUT_POSTFIX}.keras")

print(len(y_train[y_train < np.log(10)]), len(y_train[y_train >= np.log(10)]))
print(len(common_predictions), len(rare_predictions))
print(stage_one_len, stage_two_len, common_mae, rare_mae, (mae + rare_mae)/2, None if best_weight_index is None else WEIGHT_CANDIDATES[best_weight_index], None if second_stage_best_weight_index is None else WEIGHT_CANDIDATES[second_stage_best_weight_index])

# print(np.count_nonzero(common_sample_mask), np.count_nonzero(~common_sample_mask))

imbal.regression.plot_true_vs_predictions(
    y_test,
    predictions,
    title=f'SEP-E - Common MAE: {common_mae:.4f}, Rare MAE: {rare_mae:.4f}, AORE: {(mae + rare_mae)/2:.4f}',
    save_figure=f"{OUTPUT_PATH}/tvp/{DATA_PREFIX}_{'w' if VALIDATION_DATA or AE else ''}{'_validation' if VALIDATION_DATA else ''}{'_ae' if AE else ''}{'_third_last' if AE_THIRD_TO_LAST and AE else ''}{OUTPUT_POSTFIX}_tvp.png"
)

imbal.regression.tsne_visualization(
    model,
    x_test,
    y_test,
    perplexity=50,
    representation_layer_index=REPRESENTATION_LAYER_INDEX,
    save_figure=f"{OUTPUT_PATH}/tsne/{DATA_PREFIX}_{'w' if VALIDATION_DATA or AE else ''}{'_validation' if VALIDATION_DATA else ''}{'_ae' if AE else ''}{'_third_last' if AE_THIRD_TO_LAST and AE else ''}{OUTPUT_POSTFIX}_tsne.png"
)
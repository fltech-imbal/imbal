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
FIT = FitType.DECOUPLED
VALIDATION_DATA = True
AE = False
AE_THIRD_TO_LAST = False
WEIGHT_CANDIDATES = False
SINGLE_WEIGHT_ALPHA = 1

REPRESENTATION_LAYER_INDEX = -2
EARLY_STOPPING_PATIENCE = 2
EPOCHS = 10000 if VALIDATION_DATA else 200

DATA_PATH = "cleaned-dtw-SEP-EC-data"
DATA_PREFIX = 'sep_e_delta_log_normalized'
OUTPUT_PATH = "dtw-results"
USE_DELTA = True

# Will be mostly left unchanged
STRATIFY = True
BATCH_SIZE = 2048
KDE_BIN_COUNT=64
SEED = 42

"""
Load data
"""

def load_sep_ec(path_prefix):
    training_data = pd.read_csv(path_prefix + '_training.csv')
    test_data = pd.read_csv(path_prefix + '_test.csv')
    val_data = pd.read_csv(path_prefix + '_validation.csv')
    if USE_DELTA:
        training_labels = training_data.pop("delta_log_Intensity")
        val_labels = val_data.pop("delta_log_Intensity")
        test_labels = test_data.pop("delta_log_Intensity")
    else:
        training_labels = training_data.pop("Proton Intensity")
        val_labels = val_data.pop("Proton Intensity")
        test_labels = test_data.pop("Proton Intensity")
    training_data = training_data.to_numpy()
    val_data = val_data.to_numpy()
    test_data = test_data.to_numpy()
    training_labels = training_labels.to_numpy()
    val_labels = val_labels.to_numpy()
    test_labels = test_labels.to_numpy()
    return (training_data, training_labels), (val_data, val_labels), (test_data, test_labels)

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_sep_ec(
    f"{DATA_PATH}/{DATA_PREFIX}",
)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

"""
Build model
"""

if FIT == FitType.REGULAR:
    WEIGHT_CANDIDATES = False

# tf.keras.utils.set_random_seed(
#     SEED
# )

LAYER_DIMS = [128, 128, 128, 64, 64, 64, 32, 32, 32]

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

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    weighted_metrics=['mae'],
    generate_decoder_branch=AE,
    representation_layer_index=-3 if AE_THIRD_TO_LAST else -2
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

fit_function(
    x_train,
    y_train,
    sample_weight=sample_weights,
    validation_data=val_data,
    epochs=(EPOCHS, EPOCHS) if FIT == FitType.DECOUPLED else EPOCHS,
    batch_size=BATCH_SIZE,
    candidate_evaluation_sample_weight=(val_data[2][-1] if VALIDATION_DATA else sample_weights[-1]) if WEIGHT_CANDIDATES else None,
    callbacks=[keras.callbacks.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, min_delta=1e-5)] if VALIDATION_DATA else None
)

predictions = model.predict(x_test)
predictions = predictions.reshape(-1)
y_test = y_test.reshape(-1)

test_data = pd.read_csv(f"{DATA_PATH}/sep_e_log" + '_test.csv')
p_t = test_data['p_t'].to_numpy()

y_test = pd.read_csv(f"{DATA_PATH}/sep_e_log" + '_test.csv').pop("Proton Intensity").to_numpy().reshape(-1)

predictions += p_t.reshape(-1)

common_sample_mask = y_test < np.log(10)
common_predictions = predictions[common_sample_mask]
rare_predictions = predictions[~common_sample_mask]
common_labels = y_test[common_sample_mask]
rare_labels = y_test[~common_sample_mask]

mae = np.mean(np.abs(predictions - y_test))
common_mae = np.mean(np.abs(common_predictions - common_labels))
rare_mae = np.mean(np.abs(rare_predictions - rare_labels))

# model.save(f"models/{DATA_PREFIX}_{FIT.name.lower()}_{'w' if VALIDATION_DATA or AE else ''}{'_validation' if VALIDATION_DATA else ''}{'_ae' if AE else ''}{'_third_last' if AE_THIRD_TO_LAST and AE else ''}.keras")

print(np.count_nonzero(common_sample_mask), np.count_nonzero(~common_sample_mask))
imbal.regression.plot_true_vs_predictions(
    y_test,
    predictions,
    title=f'SEP-E-D - Common MAE: {common_mae:.4f}, Rare MAE: {rare_mae:.4f}, AORE: {(mae + rare_mae)/2:.4f}{f", Alpha: {[0.1*(i+1) for i in range(10)][model.best_weight_index]:.1f}" if WEIGHT_CANDIDATES else ""}',
    save_figure=f"{OUTPUT_PATH}/{DATA_PREFIX}_{FIT.name.lower()}_{'w' if VALIDATION_DATA or AE else ''}{'_validation' if VALIDATION_DATA else ''}{'_ae' if AE else ''}{'_third_last' if AE_THIRD_TO_LAST and AE else ''}.png"
)
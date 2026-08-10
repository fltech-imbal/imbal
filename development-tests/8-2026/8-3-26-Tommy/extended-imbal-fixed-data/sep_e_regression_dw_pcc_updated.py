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
AE = True
AE_THIRD_TO_LAST = False
WEIGHT_CANDIDATES = True
SINGLE_WEIGHT_ALPHA = 1

REPRESENTATION_LAYER_INDEX = -2
EARLY_STOPPING_PATIENCE = 200
EPOCHS = 10000 if VALIDATION_DATA else 200

DATA_PATH = 'cleaned-dtw-SEP-EC-data'
DATA_PREFIX = 'sep_ec_log_normalized'
USE_DELTA = False

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

print(f"Train: {x_train.shape}")
print(f'Validation: {x_val.shape}')
print(f'Test: {x_test.shape}')

print(len(y_train[y_train < np.log(10)]), len(y_train[y_train >= np.log(10)]))

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

def pcc(y_true, y_pred):
    y_true = tf.reshape(y_true, tf.shape(y_pred))

    y_true_centered = y_true - tf.reduce_mean(y_true)
    y_pred_centered = y_pred - tf.reduce_mean(y_pred)

    return 1 - (
        tf.reduce_sum(y_true_centered * y_pred_centered) /
        (tf.norm(y_true_centered) * tf.norm(y_pred_centered))
    )

@keras.saving.register_keras_serializable()
def loss_fn(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true, y_pred) + 0.5*pcc(y_true, y_pred)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=loss_fn,
    weighted_metrics=['mae'],
    generate_decoder_branch=AE,
    representation_layer_index=-3 if AE_THIRD_TO_LAST else -2
)

if FIT == FitType.DECOUPLED:
    model.override_second_stage_fit_parameters(
        callbacks=[keras.callbacks.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, min_delta=1e-4)] if VALIDATION_DATA else None
    )

"""
Generate sample densities
"""


fit_function = model.fit
if FIT == FitType.BALANCED:
    fit_function = model.balanced_fit
if FIT == FitType.DECOUPLED:
    fit_function = model.rRT_fit

def denseweight(densities, alpha=1, epsilon=1e-3):
    min_density = np.min(densities)
    max_density = np.max(densities)
    normalized_densities = (densities - min_density) / (max_density - min_density)

    if isinstance(alpha, list):
        return np.array([np.maximum(1-a*normalized_densities, epsilon) for a in alpha])
    else:
        return np.maximum(1-alpha*normalized_densities, epsilon)

candidate_eval_weights = None

if VALIDATION_DATA:
    kde_bandwidth = imbal.regression.fit_kde(
        y_train,
        bin_count=KDE_BIN_COUNT
    )

    sample_densities = imbal.regression.get_sample_densities(
        y_train,
        kde_bandwidth,
    )
    sample_weights = denseweight(sample_densities, alpha=[0.25*(i+1) for i in range(10)] if WEIGHT_CANDIDATES else 1)
    val_densities = imbal.regression.get_sample_densities(
        y_val,
        kde_bandwidth,
        distribution=y_train
    )
    w_val = denseweight(val_densities, alpha=[0.25*(i+1) for i in range(10)] if WEIGHT_CANDIDATES else 1)
    val_data = (x_val, y_val, w_val)
    candidate_eval_weights = imbal.regression.reciprocal_importance(val_densities)
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
    sample_weights = denseweight(sample_densities)

    val_data = None
    candidate_eval_weights = imbal.regression.reciprocal_importance(sample_densities)

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
    candidate_evaluation_sample_weight=candidate_eval_weights if WEIGHT_CANDIDATES else None,
    callbacks=[keras.callbacks.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, min_delta=1e-5)] if VALIDATION_DATA else None
)

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

model.save(f"models/{DATA_PATH}_{FIT.name.lower()}_{'w' if VALIDATION_DATA or AE else ''}{'_validation' if VALIDATION_DATA else ''}{'_ae' if AE else ''}{'_third_last' if AE_THIRD_TO_LAST and AE else ''}_denseweight_pcc.keras")


imbal.regression.plot_true_vs_predictions(
    y_test,
    predictions,
    title=f'Common MAE: {common_mae:.4f}, Rare MAE: {rare_mae:.4f}, AORE: {(mae + rare_mae)/2:.4f}{f", Alpha: {[0.25*(i+1) for i in range(10)][model.best_weight_index]:.2f}" if WEIGHT_CANDIDATES else ""}',
    save_figure=f"dtw-results/{DATA_PATH}_{FIT.name.lower()}_{'w' if VALIDATION_DATA or AE else ''}{'_validation' if VALIDATION_DATA else ''}{'_ae' if AE else ''}{'_third_last' if AE_THIRD_TO_LAST and AE else ''}_denseweight_pcc.png"
)
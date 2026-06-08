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

LEARNING_RATE = 1e-4
FIT = FitType.BALANCED
AE = True
VALIDATION_DATA = True
VALIDATION_SPLIT = 0.15

REPRESENTATION_LAYER_INDEX = -2
INCLUDE_CME_DATA = True
DETERMINE_BEST_IMPORTANCE = False
K_FOLD_EPOCHS = False
K_FOLD_METRIC = 'val_loss'
GEN_OUTPUT = True
EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 20
EPOCHS = 50

DATA_PATH = 'sep_e_no_electron_log_normalized'
OUTPUT_PATH = 'out' + ('' if INCLUDE_CME_DATA else '-no-cme')

# Will be mostly left unchanged
STRATIFY = True
BATCH_SIZE = 512
KDE_BIN_COUNT=64
SEED = 42

"""
Load data
"""

def load_sep_ec(path_prefix):
    training_data = pd.read_csv(path_prefix + '_training.csv')
    test_data = pd.read_csv(path_prefix + '_test.csv')
    training_labels = training_data.pop("delta_log_Intensity")
    test_labels = test_data.pop("delta_log_Intensity")
    training_data = training_data.to_numpy()
    test_data = test_data.to_numpy()
    training_labels = training_labels.to_numpy()
    test_labels = test_labels.to_numpy()
    return (training_data, training_labels), (test_data, test_labels)

(x_train, y_train), (x_test, y_test) = load_sep_ec(
    f"SEP-E/{DATA_PATH}",
)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

"""
Build model
"""

tf.keras.utils.set_random_seed(
    SEED
)

LAYER_DIMS = [128, 128, 128, 64, 64, 64, 32, 32, 32]

inputs = keras.Input(shape=(x_train.shape[1],))

x = inputs
for index, num_units in enumerate(LAYER_DIMS):
    if index == len(LAYER_DIMS) - 1:
        x = layers.Flatten()(x)
    x = layers.Dense(num_units, activation='relu')(x)

outputs = layers.Dense(1)(x)

model = imbal.regression.Model(inputs=inputs, outputs=outputs, name="SEP_EC")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    weighted_metrics=['mae'],
    generate_decoder_branch=AE
)

"""
Generate sample densities
"""
kde_bandwidth = imbal.regression.fit_kde(
    y_train,
    bin_count=KDE_BIN_COUNT
)
# imbal.regression.plot_kde_1d(
#     y_train,
#     kde_bandwidth,
#     bin_count=KDE_BIN_COUNT
# )
densities = imbal.regression.get_sample_densities(
    y_train,
    kde_bandwidth,
)
sample_weights = imbal.regression.reciprocal_importance(densities)

fit_function = model.fit
if FIT == FitType.BALANCED:
    fit_function = model.balanced_fit
if FIT == FitType.DECOUPLED:
    fit_function = model.rRT_fit

if VALIDATION_DATA:
    (x_train, y_train, sample_weights), (x_val, y_val, w_val) = imbal.regression.split(x_train, y_train, sample_weights, test_size=VALIDATION_SPLIT)
    train_common_mask = (y_train > -0.5) & (y_train < 0.5)
    val_common_mask = (y_val > -0.5) & (y_val < 0.5)
    print(len(y_train[train_common_mask]), len(y_train[~train_common_mask]))
    print(len(y_val[val_common_mask]), len(y_val[~val_common_mask]))
    val_data = (x_val, y_val, w_val)
else:
    val_data = None


fit_function(
    x_train,
    y_train,
    sample_weight=sample_weights,
    validation_data=val_data,
    epochs=EPOCHS
)

predictions = model.predict(x_test)
predictions = predictions.reshape(-1)
y_test = y_test.reshape(-1)

common_predictions_mask = (predictions > -0.5) & (predictions < 0.5)
common_predictions = predictions[common_predictions_mask]
rare_predictions = predictions[~common_predictions_mask]
common_labels = y_test[common_predictions_mask]
rare_labels = y_test[~common_predictions_mask]

common_mae = np.mean(np.abs(common_predictions - common_labels))
rare_mae = np.mean(np.abs(rare_predictions - rare_labels))

imbal.regression.plot_true_vs_predictions(
    y_test,
    predictions,
    title=f'Common MAE: {common_mae:.4f}, Rare MAE: {rare_mae:.4f}',
    save_figure=f"results/{DATA_PATH}_{FIT.name.lower()}_{'w' if VALIDATION_DATA or AE else ''}{'_validation' if VALIDATION_DATA else ''}{'_ae' if AE else ''}.png"
)
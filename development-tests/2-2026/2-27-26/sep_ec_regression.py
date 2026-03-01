"""
Import packages
"""
import imbal
import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np
import time
import tools
from tools import FitType

"""
Set script parameters
"""

LEARNING_RATE = 2e-4
FIT = FitType.DECOUPLED
AE = True
REPRESENTATION_LAYER_INDEX = -4
DETERMINE_BEST_IMPORTANCE = False
INCLUDE_CME_DATA = True
GEN_OUTPUT = True

OUTPUT_PATH = 'out' + ('' if INCLUDE_CME_DATA else '-no-cme')

# Will be mostly left unchanged
STRATIFY = True
BATCH_SIZE = 512
KDE_BIN_COUNT=64
EPOCHS = 10000
SEED = 42
EARLY_STOPPING_PATIENCE = 20

"""
Load data
"""

(x_train, y_train), (x_test, y_test) = tools.load_sep_ec(
    "../../../tutorials/data/SEP-EC",
    include_cme=INCLUDE_CME_DATA
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

input_shape = x_train.shape[1:]
inputs = keras.Input(shape=input_shape)
x = layers.Dense(18, activation='relu')(inputs)
x = layers.Dense(9, activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(6, activation='relu')(x)
x = layers.Flatten()(x)
output = layers.Dense(1)(x)
model = imbal.regression.Model(inputs=inputs, outputs=output)
model.summary()

model.compile(
    loss='mse',
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["mse"],
    generate_decoder_branch=AE,
    representation_layer_index=REPRESENTATION_LAYER_INDEX
)

"""
Generate sample densities
"""

kde_bandwidth = imbal.regression.fit_kde(
    y_train,
    bin_count=KDE_BIN_COUNT
)
imbal.regression.plot_kde_1d(
    y_train,
    kde_bandwidth,
    bin_count=KDE_BIN_COUNT
)
densities = imbal.regression.get_sample_densities(
    y_train,
    kde_bandwidth,
)

"""
Train model
"""
# Determine fit function to use
fit_function = model.fit
if FIT == FitType.BALANCED:
    fit_function = model.balanced_fit
if FIT == FitType.DECOUPLED:
    fit_function = model.rRT_fit

# Determine weights to use based on fit function
weights = np.ones(x_train.shape[0])
if FIT != FitType.REGULAR:
    densities = imbal.regression.get_sample_densities(
        y_train,
        kde_bandwidth
    )
    weights = imbal.regression.generate_sample_weights(densities)

# Necessary for early stopping during rRT_fit
model.override_second_stage_fit_parameters(
    epochs=EPOCHS,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
    ]
)

if DETERMINE_BEST_IMPORTANCE:
    possible_weights = imbal.regression.reciprocal_importance(densities, (0, 1))
    best_weights, best_loss = tools.find_best_performance(
        model,
        model_fit_function=fit_function,
        sample_weight_contenders=possible_weights,
        early_stopping_parameters={
            "patience": EARLY_STOPPING_PATIENCE,
            "restore_best_weights": True
        },
        additional_fit_parameters={
            "x":x_train,
            "y":y_train,
            "stratify_batches":STRATIFY,
            "validation_split":0.2,
            "batch_size":BATCH_SIZE,
            "epochs":EPOCHS
        }
    )

    print("BEST VAL LOSS:", best_loss)
    for i in range(len(possible_weights)):
        if (possible_weights[i] == best_weights).all():
            print("Best alpha:", i/10)
    weights = best_weights

start = time.time()
history = fit_function(
    x_train,
    y_train,
    stratify_batches=STRATIFY,
    validation_split=0.2,
    sample_weight=weights,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
    ]
)
end = time.time()

if FIT == FitType.DECOUPLED:
    one, two = history
    print('Stage lengths:')
    print(len(one.epoch), len(two.epoch))

"""
Evaluate model
"""
print('EXECUTION TIME:', end - start)
print('Evaluating model...')
model.evaluate(x_test, y_test)
predictions = model.predict(x_test)

imbal.regression.tsne_visualization(
    model,
    x_test,
    y_test,
    representation_layer_index=REPRESENTATION_LAYER_INDEX,
    save_figure=f'{OUTPUT_PATH}/tsne_visualization-{FIT.value}-ae-{AE}-rep{REPRESENTATION_LAYER_INDEX}.png' if GEN_OUTPUT else None,
)

tools.plot_true_vs_predictions(
    y_test,
    predictions,
    save_figure=f'{OUTPUT_PATH}/regression-true-pred-{FIT.value}-ae-{AE}-rep{REPRESENTATION_LAYER_INDEX}.png' if GEN_OUTPUT else None,
)


def pearson_correlation_coefficient(y, y_hat):
    return np.corrcoef(y, y_hat)[0, 1]

predictions = predictions.reshape(-1)
rare_mask = (y_test < -1) | (y_test > 1)
rare_y_test = y_test[rare_mask]
rare_predictions = predictions[rare_mask]
print(y_test.shape)
print(rare_y_test.shape)
print(predictions.shape)
print(rare_predictions.shape)
mae = np.mean(np.abs(predictions - y_test))
mae_r = np.mean(np.abs(rare_predictions - rare_y_test))
pcc = pearson_correlation_coefficient(y_test, predictions)
pcc_r = pearson_correlation_coefficient(rare_y_test, rare_predictions)

print('MAE:', f'{mae:.5f}')
print('MAE_R:', f'{mae_r:.5f}')
print('AORE:', f'{(mae_r + mae)/2:.5f}')

print('PCC:', f'{pcc:.5f}')
print('PCC_R:', f'{pcc_r:.5f}')
print('AORC:', f'{(pcc_r + pcc)/2:.5f}')

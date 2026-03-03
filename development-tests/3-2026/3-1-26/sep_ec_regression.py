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
from tools import FitType, AORE
import daniel_tools
from daniel_tools import fit_k_folds_modular
"""
Set script parameters
"""

# tf.config.run_functions_eagerly(True)

LEARNING_RATE = 2e-4
FIT = FitType.BALANCED
AE = False
REPRESENTATION_LAYER_INDEX = -2
INCLUDE_CME_DATA = True
DETERMINE_BEST_IMPORTANCE = True
K_FOLD_EPOCHS = True
K_FOLD_METRIC = 'val_aore'
GEN_OUTPUT = True
EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 20
EPOCHS = 224

OUTPUT_PATH = 'out' + ('' if INCLUDE_CME_DATA else '-no-cme')

# Will be mostly left unchanged
STRATIFY = True
BATCH_SIZE = 512
KDE_BIN_COUNT=64
SEED = 42

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

model = tools.build_sep_ec_model(input_dim=x_train.shape[1])
model.summary()

model.compile(
    loss='mse',
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["mse", AORE()],
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
# imbal.regression.plot_kde_1d(
#     y_train,
#     kde_bandwidth,
#     bin_count=KDE_BIN_COUNT
# )
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
    # weights = imbal.regression.generate_sample_weights(densities)
    weights = imbal.regression.dense_weight(densities, alpha=0.1)
# Necessary for early stopping during rRT_fit
model.override_second_stage_fit_parameters(
    callbacks=[
        keras.callbacks.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
    ]
)

if K_FOLD_EPOCHS:
    print('Performing k-fold validation...')

    if FIT == FitType.DECOUPLED:
        model.fit(
            x_train,
            y_train,
            stratify_batches=STRATIFY,
            validation_split=0.2,
            batch_size=BATCH_SIZE,
            epochs=118
        )

    def k_fold(weights):

        early_stop = keras.callbacks.EarlyStopping(
            monitor=K_FOLD_METRIC,
            patience=50,
            mode="min",
            restore_best_weights=True
        )

        params = (daniel_tools.RegularFitParams(
            x=x_train,
            y=y_train,
            sample_weight=weights,
            batch_size=BATCH_SIZE,
            shuffle=True,  # keep your intended behavior
            callbacks=[early_stop],  # NOTE: only pass the EarlyStopping here if you want it cloned per fold
        ) if FIT == FitType.REGULAR else (
        daniel_tools.BalancedFitParams(
            x=x_train,
            y=y_train,
            sample_weight=weights,
            batch_size=BATCH_SIZE,
            shuffle=True,  # keep your intended behavior
            callbacks=[early_stop],  # NOTE: only pass the EarlyStopping here if you want it cloned per fold
        ) if FIT == FitType.BALANCED else
        daniel_tools.DecoupledFitStageTwoParams(
            x=x_train,
            y=y_train,
            sample_weight=weights,
            batch_size=BATCH_SIZE,
            shuffle=True,  # keep your intended behavior
            callbacks=[early_stop],  # NOTE: only pass the EarlyStopping here if you want it cloned per fold
        )))


        return fit_k_folds_modular(
            model,
            x=x_train,
            y=y_train,
            strategy=(
                daniel_tools.RegularFitStrategy() if FIT == FitType.REGULAR else (
                daniel_tools.BalancedFitStrategy() if FIT == FitType.BALANCED else
                daniel_tools.DecoupledFitStageTwoStrategy())
            ),
            params=params,
            batch_size=BATCH_SIZE,
            num_folds=5,
            shuffle=True,
            seed=SEED,
            mode=imbal.util.backend.ModelType.REGRESSION

        )
    if DETERMINE_BEST_IMPORTANCE:
        possible_weights = np.concatenate([
            imbal.regression.reciprocal_importance(densities, (0, 1)),
            imbal.regression.dense_weight(densities, (0.1, 2), steps=19)
            # [imbal.regression.reciprocal_importance(densities, alpha=0)],
            # [imbal.regression.reciprocal_importance(densities, alpha=1)],
            # [imbal.regression.dense_weight(densities, alpha=1)]
        ])
        for contender in possible_weights:
            print(np.min(contender), np.max(contender))

        best_metric = None
        best_epochs = None
        best_index = None
        initial_weights = model.get_weights()
        for i, weight_candidate in enumerate(possible_weights):
            print(f'Examining candidate weights... [{i+1}/{len(possible_weights)}]')
            k_fold_epochs, average_metric = k_fold(weight_candidate)
            if best_metric is None or average_metric < best_metric:
                best_metric = average_metric
                best_epochs = k_fold_epochs
                best_index = i
        print('Best metric:', best_metric)
        print('Index:', best_index)
        EPOCHS = best_epochs
        weights = possible_weights[best_index]
    else:
        EPOCHS, _ = k_fold(weights)

    print('\nEpochs:', EPOCHS, '\n')


if not K_FOLD_EPOCHS and DETERMINE_BEST_IMPORTANCE:
    possible_weights = np.concatenate([
        imbal.regression.reciprocal_importance(densities, (0, 1)),
        imbal.regression.dense_weight(densities, (0.1, 2), steps=19)])

    best_weights, best_loss = tools.find_best_performance(
        model,
        model_fit_function=fit_function,
        sample_weight_candidates=possible_weights,
        early_stopping_parameters={
            "patience": EARLY_STOPPING_PATIENCE,
            "restore_best_weights": True
        },
        additional_fit_parameters={
            "x":x_train,
            "y":y_train,
            "stratify_batches":STRATIFY,
            "validation_split":0.2,
            "batch_size":BATCH_SIZE
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
    sample_weight=weights,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=(
        [keras.callbacks.EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )] if EARLY_STOPPING and not K_FOLD_EPOCHS else [],
    )
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
rare_mask = (y_test < -0.5) | (y_test > 0.5)
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

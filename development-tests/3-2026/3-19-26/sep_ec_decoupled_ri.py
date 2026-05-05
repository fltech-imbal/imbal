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
from tools import FitType, AORE, TrainingPhaseManager, IsTraining, mdi_importance

"""
Set script parameters
"""

LEARNING_RATE = 2e-4
FIT = FitType.DECOUPLED
AE = True
REPRESENTATION_LAYER_INDEX = -2
USE_WPCC = True
EPOCHS = 514
DENSITY_TO_WEIGHT_FUNCTION = imbal.regression.dense_weight
DENSITY_TO_WEIGHT_KWARGS = {
    'alpha' : 1.9
}
FREEZE_SECOND_STAGE_EXTENDED = True

INCLUDE_CME_DATA = True
DETERMINE_BEST_IMPORTANCE = True
K_FOLD_EPOCHS = True
K_FOLD_METRIC = 'val_loss'
GEN_OUTPUT = True
EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 100
MIN_EPOCHS_BEFORE_STOP = 250

OUTPUT_PATH = 'out' + ('' if INCLUDE_CME_DATA else '-no-cme')

# Will be mostly left unchanged
STRATIFY = True
BATCH_SIZE = 2048
KDE_BIN_COUNT=64
SEED = 42

"""
Load data
"""

x_train, y_train, normalization_factors = tools.load_sep_ec(
    "SEP-EC/training",
    include_cme=INCLUDE_CME_DATA,
    return_normalization_factors=True,
)
x_test, y_test = tools.load_sep_ec(
    "SEP-EC/testing",
    include_cme=INCLUDE_CME_DATA,
    normalization_factors=normalization_factors
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
print(model.layers)
representation_layer = model.get_layer(index=-2)
print(representation_layer)
model.summary()

pm = TrainingPhaseManager()

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
if FIT == FitType.DECOUPLED or FIT == FitType.EXTENDED:
    fit_function = model.rRT_fit

# Determine weights to use based on fit function
weights = np.ones(x_train.shape[0])
if FIT != FitType.REGULAR:
    densities = imbal.regression.get_sample_densities(
        y_train,
        kde_bandwidth
    )
    # weights = imbal.regression.generate_sample_weights(densities)
    weights = DENSITY_TO_WEIGHT_FUNCTION(densities, **DENSITY_TO_WEIGHT_KWARGS)
# Necessary for early stopping during rRT_fit
model.override_second_stage_fit_parameters(
    callbacks=[
        tools.DelayedEarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            stop_after_epoch=MIN_EPOCHS_BEFORE_STOP
        ),
        IsTraining(pm)
    ]
)

if K_FOLD_EPOCHS:
    print('Performing k-fold validation...')

    if FIT == FitType.DECOUPLED or FIT == FitType.EXTENDED:
        train_ones_dict = tools.map_labels_to_importance_weights(y_train, np.ones(len(y_train)))
        model.compile(
            loss=lambda y_true, y_pred: tools.mse_wpcc(
                y_true, y_pred,
                phase_manager=pm,
                lambda_factor=0.5 if USE_WPCC else 0,
                train_mse_weight_dict=train_ones_dict,
                train_pcc_weight_dict=train_ones_dict,
            ),
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            metrics=["mse", AORE()],
            generate_decoder_branch=AE,
            representation_layer_index=REPRESENTATION_LAYER_INDEX
        )

        model.fit(
            x_train,
            y_train,
            stratify_batches=STRATIFY,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS
        )

        _, rep_layer_index = imbal.util.get_representation_layer_index(model, REPRESENTATION_LAYER_INDEX)
        if FIT == FitType.DECOUPLED or FREEZE_SECOND_STAGE_EXTENDED:
            for index, layer in enumerate(model.layers):
                if index <= rep_layer_index:
                    layer.trainable = False

        if FIT == FitType.EXTENDED:
            extra_layer = layers.Dense(64)(representation_layer.output)
            new_output = layers.Dense(1)(extra_layer)
            new_model = imbal.regression.Model(inputs=model.input, outputs=new_output, name="SEP_EC_extended")
            new_model.summary()

            new_model.compile(
                loss='mse',
                optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                metrics=["mse", AORE()]
            )

            model = new_model


        fit_function = model.balanced_fit

    def k_fold(weight_function, weight_alpha):

        maes = []
        mae_rs = []
        aores = []
        best_epochs = []
        best_loss = []

        initial_weights = model.get_weights()

        for i in range(4):
            x_sub_train, y_sub_train = tools.load_sep_ec(
                f"SEP-EC/fold{i}/subtraining",
                include_cme=INCLUDE_CME_DATA,
                normalization_factors=normalization_factors
            )
            x_val, y_val = tools.load_sep_ec(
                f"SEP-EC/fold{i}/validation",
                include_cme=INCLUDE_CME_DATA,
                normalization_factors=normalization_factors
            )


            if FIT != FitType.REGULAR:
                
                densities_sub_train = imbal.regression.get_sample_densities(
                    y_sub_train,
                    kde_bandwidth,
                    distribution=y_train
                )
                weights_sub_train = weight_function(densities_sub_train, alpha=weight_alpha)

                densities_val = imbal.regression.get_sample_densities(
                    y_val,
                    kde_bandwidth,
                    distribution=y_train
                )
                weights_val = weight_function(densities_val, alpha=weight_alpha)
            else:
                weights_sub_train = np.ones(x_sub_train.shape[0])
                weights_val = np.ones(x_val.shape[0])

            print('weight shape:', weights_sub_train.shape)
            train_weight_dict = tools.map_labels_to_importance_weights(y_sub_train, weights_sub_train)
            train_ones_dict = tools.map_labels_to_importance_weights(y_sub_train, np.ones(len(y_sub_train)))

            val_weight_dict = tools.map_labels_to_importance_weights(y_val, weights_val)
            val_ones_dict = tools.map_labels_to_importance_weights(y_val, np.ones(len(y_val)))

            model.compile(
                loss=lambda y_true, y_pred: tools.mse_wpcc(
                    y_true, y_pred,
                    phase_manager=pm,
                    lambda_factor=0.5 if USE_WPCC else 0,
                    train_mse_weight_dict=train_weight_dict,
                    train_pcc_weight_dict=train_ones_dict,
                    val_mse_weight_dict=val_weight_dict,
                    val_pcc_weight_dict=val_ones_dict,
                ),
                optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                metrics=["mse", AORE()],
                generate_decoder_branch=False,
                representation_layer_index=REPRESENTATION_LAYER_INDEX
            )

            model.set_weights(initial_weights)
            model.reset_metrics()

            history = fit_function(
                x_sub_train,
                y_sub_train,
                sample_weight=weights_sub_train,
                validation_data=(x_val, y_val, weights_val),
                stratify_batches=STRATIFY,
                batch_size=BATCH_SIZE,
                epochs=100000,
                callbacks=[
                    tools.DelayedEarlyStopping(
                        patience=EARLY_STOPPING_PATIENCE,
                        restore_best_weights=True,
                        monitor=K_FOLD_METRIC,
                        stop_after_epoch=MIN_EPOCHS_BEFORE_STOP,
                        mode="min"
                    ),
                    IsTraining(pm)
                ]

            )

            val_losses = history.history['val_loss']
            if val_losses:
                best_loss_index = int(np.argmin(val_losses))
                print('Found of minimum val_loss:', best_loss_index+1)
                best_loss.append(val_losses[best_loss_index])
                best_epochs.append(best_loss_index + 1)

            test_predictions = model.predict(x=x_val)
            test_predictions = test_predictions.reshape(-1)
            y_val = y_val.reshape(-1)

            rare_mask = (y_val < -0.5) | (y_val > 0.5)
            rare_y_val = y_val[rare_mask]
            rare_predictions = test_predictions[rare_mask]

            mae = np.mean(np.abs(test_predictions - y_val))
            mae_r = np.mean(np.abs(rare_y_val - rare_predictions))
            aore = (mae + mae_r) / 2
            maes.append(mae)
            mae_rs.append(mae_r)
            aores.append(aore)

        avg_best_epoch = int(np.rint(np.mean(best_epochs))) if best_epochs else 1
        avg_best_epoch = max(1, avg_best_epoch)
        print("Best epoch (average):", avg_best_epoch)

        model.set_weights(initial_weights)

        return {
            "epoch": avg_best_epoch,
            "val_loss": np.mean(best_loss),
            "mae" : np.mean(maes),
            "mae_r": np.mean(mae_rs),
            "aore": np.mean(aores),
        }

    if DETERMINE_BEST_IMPORTANCE:
        possible_weights = [
            
            [imbal.regression.reciprocal_importance, 0], [imbal.regression.reciprocal_importance, 0.1],
            [imbal.regression.reciprocal_importance, 0.2], [imbal.regression.reciprocal_importance, 0.3],
            [imbal.regression.reciprocal_importance, 0.4], [imbal.regression.reciprocal_importance, 0.5],
            [imbal.regression.reciprocal_importance, 0.6], [imbal.regression.reciprocal_importance, 0.7],
            [imbal.regression.reciprocal_importance, 0.8], [imbal.regression.reciprocal_importance, 0.9],
            [imbal.regression.reciprocal_importance, 1.0],

            # [imbal.regression.dense_weight, 0.1],
            # [imbal.regression.dense_weight, 0.2], [imbal.regression.dense_weight, 0.3],
            # [imbal.regression.dense_weight, 0.4], [imbal.regression.dense_weight, 0.5],
            # [imbal.regression.dense_weight, 0.6], [imbal.regression.dense_weight, 0.7],
            # [imbal.regression.dense_weight, 0.8], [imbal.regression.dense_weight, 0.9],
            # [imbal.regression.dense_weight, 1.0], [imbal.regression.dense_weight, 1.1],
            # [imbal.regression.dense_weight, 1.2], [imbal.regression.dense_weight, 1.3],
            # [imbal.regression.dense_weight, 1.4], [imbal.regression.dense_weight, 1.5],
            # [imbal.regression.dense_weight, 1.6], [imbal.regression.dense_weight, 1.7],
            # [imbal.regression.dense_weight, 1.8], [imbal.regression.dense_weight, 1.9],
            # [imbal.regression.dense_weight, 2.0],

            # [mdi_importance, 0.1],
            # [mdi_importance, 0.2], [mdi_importance, 0.3],
            # [mdi_importance, 0.4], [mdi_importance, 0.5],
            # [mdi_importance, 0.6], [mdi_importance, 0.7],
            # [mdi_importance, 0.8], [mdi_importance, 0.9],
            # [mdi_importance, 1.0], [mdi_importance, 1.1],
            # [mdi_importance, 1.25], [mdi_importance, 1.4],
            # [mdi_importance, 1.66], [mdi_importance, 2],
            # [mdi_importance, 2.5], [mdi_importance, 3.33],
            # [mdi_importance, 5], [mdi_importance, 10]
        ]
        # for contender in possible_weights:
        #     print(np.min(contender), np.max(contender))

        k_fold_metrics = []
        best_metric = None
        best_epochs = None
        best_index = None
        initial_weights = model.get_weights() 
        for i, weight_candidate in enumerate(possible_weights):
            print(f'Examining candidate weights... [{i+1}/{len(possible_weights)}]')
            fold_results = k_fold(weight_candidate[0], weight_candidate[1])
            average_metric = fold_results['aore']
            k_fold_epochs = fold_results['epoch']
            k_fold_metrics.append(fold_results)
            if best_metric is None or average_metric > best_metric:
                best_metric = average_metric
                best_epochs = k_fold_epochs
                best_index = i
            print(fold_results)
        print('Best metric:', best_metric)
        print('Index:', best_index)
        EPOCHS = best_epochs
        weights = possible_weights[best_index]

        print('\nEpochs:', EPOCHS, '\n')
        headers = ['alpha', 'MAE', 'MAE_r', 'AORE', 'val_loss', 'epochs']
        weight_label = [
                        'instance', 'reciprocal, 0.1', 'reciprocal, 0.2', 'reciprocal, 0.3', 'reciprocal, 0.4',
                        'reciprocal, 0.5', 'reciprocal, 0.6', 'reciprocal, 0.7', 'reciprocal, 0.8', 'reciprocal, 0.9',
                        'reciprocal, 1.0',
                        # 'denseweight, 0.1', 'denseweight, 0.2', 'denseweight, 0.3', 'denseweight, 0.4',
                        # 'denseweight, 0.5', 'denseweight, 0.6', 'denseweight, 0.7', 'denseweight, 0.8', 'denseweight, 0.9',
                        # 'denseweight, 1.0', 'denseweight, 1.1', 'denseweight, 1.2', 'denseweight, 1.3', 'denseweight, 1.4',
                        # 'denseweight, 1.5', 'denseweight, 1.6', 'denseweight, 1.7', 'denseweight, 1.8', 'denseweight, 1.9',
                        # 'denseweight, 2.0'
                        # 'mdi, 0.1', 'mdi, 0.2', 'mdi, 0.3', 'mdi, 0.4',
                        # 'mdi, 0.5', 'mdi, 0.6', 'mdi, 0.7', 'mdi, 0.8', 'mdi, 0.9',
                        # 'mdi, 1.0', 'mdi, 1.1', 'mdi, 1.25', 'mdi, 1.4', 'mdi, 1.66',
                        # 'mdi, 2', 'mdi, 2.5', 'mdi, 3.33', 'mdi, 5', 'mdi, 10',
        ]
        header_format_string = "{:<20}{:<10}{:<10}{:<10}{:<10}{:<10}"
        format_string = "{:<20}{:<10.3}{:<10.3}{:<10.3}{:<10.3}{:<10}"
        print(header_format_string.format(*headers))
        print("-" * 80)
        for i, row in enumerate(k_fold_metrics):
            extracted_row = [weight_label[i], row['mae'], row['mae_r'], row['aore'], row['val_loss'], row['epoch']]
            print(format_string.format(*extracted_row))
    else:
        info = k_fold(imbal.regression.reciprocal_importance, 0)
        print(info['epoch'])


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

train_weight_dict = tools.map_labels_to_importance_weights(y_train, weights)
train_ones_dict = tools.map_labels_to_importance_weights(y_train, np.ones(len(y_train)))

model.compile(
    loss=lambda y_true, y_pred: tools.mse_wpcc(
        y_true, y_pred,
        phase_manager=pm,
        lambda_factor=0.5 if USE_WPCC else 0,
        train_mse_weight_dict=train_weight_dict,
        train_pcc_weight_dict=train_ones_dict,
    ),
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["mse", AORE()],
    generate_decoder_branch=AE,
    representation_layer_index=REPRESENTATION_LAYER_INDEX
)

start = time.time()
if FIT == FitType.EXTENDED:
    model.fit(
        x_train,
        y_train,
        stratify_batches=STRATIFY,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS[0],
    )

    if FREEZE_SECOND_STAGE_EXTENDED:
        for layer in model.layers:
            layer.trainable = False

    extra_layer = layers.Dense(64)(representation_layer.output)
    new_output = layers.Dense(1)(extra_layer)
    new_model = imbal.regression.Model(inputs=model.input, outputs=new_output, name="SEP_EC_extended")
    new_model.summary()

    new_model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["mse", AORE()]
    )

    history = new_model.balanced_fit(
        x_train,
        y_train,
        stratify_batches=STRATIFY,
        sample_weight=weights,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS[1],
    )
else:
    history = fit_function(
        x_train,
        y_train,
        stratify_batches=STRATIFY,
        sample_weight=weights,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[
            IsTraining(pm)
        ],

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
    save_figure=f'{OUTPUT_PATH}/tsne_visualization-{FIT.value}-ae-{AE}-rep{REPRESENTATION_LAYER_INDEX}-weights-{DENSITY_TO_WEIGHT_FUNCTION.__name__}-{DENSITY_TO_WEIGHT_KWARGS["alpha"]}-frozen-{FREEZE_SECOND_STAGE_EXTENDED}-wpcc-{USE_WPCC}.png' if GEN_OUTPUT else None,
)

tools.plot_true_vs_predictions(
    y_test,
    predictions,
    save_figure=f'{OUTPUT_PATH}/regression-true-pred-{FIT.value}-ae-{AE}-rep{REPRESENTATION_LAYER_INDEX}-weights-{DENSITY_TO_WEIGHT_FUNCTION.__name__}-{DENSITY_TO_WEIGHT_KWARGS["alpha"]}-frozen-{FREEZE_SECOND_STAGE_EXTENDED}-wpcc-{USE_WPCC}.png' if GEN_OUTPUT else None,
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

temp = AORE()
temp.update_state(y_test, predictions)
print(temp.result())

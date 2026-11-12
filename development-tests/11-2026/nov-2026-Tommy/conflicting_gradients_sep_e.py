import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import sys
import imbal
from gradient_conflict_tools import *
from tools import load_sep_ec_data

def run_model(
    LEARNING_RATE = 5e-5,
    WEIGHT_CANDIDATES = [0.1, 0.3, 0.5, 0.7],
    SINGLE_WEIGHT_ALPHA = 1,

    DATA_PATH = "cleaned-dtw-SEP-EC-data",
    DATA_PREFIX = 'sep_e_log_normalized',
    MODEL_OUTPUT_PATH = "models-sep-e",
    OUTPUT_PATH='results-sep-e',
    UNIT_REPRESENTATIONS = False,
    NONLINEAR_REGRESSOR = False,
    LEARNABLE_RATIO = False,
    OUTPUT_POSTFIX = '_tempdelete',

    REPRESENTATION_LAYER_INDEX = -2,
    EARLY_STOPPING_PATIENCE = 100,
    EPOCHS = 10000,
    BATCH_SIZE = 2048,
    KDE_BIN_COUNT=64,

    CONFLICT_REDUCTION_METHOD=None,
    GRADIENT_BINNING_METHOD='label',
    NUM_CLASSES = 3,
    NORMALIZE_GRADIENTS_BEFORE_REDUCTION=False,
    EARLY_STOPPING_DELTA=0
):



    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_sep_ec_data(
        f"{DATA_PATH}/{DATA_PREFIX}",
    )

    print(x_train.shape)
    print(x_test.shape)
    print(x_val.shape)

    y_train_pseudo_class = np.digitize(y_train, bins=[-np.log(10), np.log(10)])
    y_test_pseudo_class = np.digitize(y_test, bins=[-np.log(10), np.log(10)])
    y_val_pseudo_class = np.digitize(y_val, bins=[-np.log(10), np.log(10)])

    """
    Build model
    """

    LAYER_DIMS = [128, 128, 128, 64, 64, 64, 32, 32, 32]
    # LAYER_DIMS = [128, 64, 64, 32, 32]

    inputs = keras.Input(shape=(x_train.shape[1],))

    x = inputs
    for index, num_units in enumerate(LAYER_DIMS):

        if index == len(LAYER_DIMS) - 1 and UNIT_REPRESENTATIONS:
            x = layers.Dense(num_units)(x)
            x = layers.UnitNormalization()(x)
        else:
            x = layers.Dense(num_units, activation='relu')(x)

    if NONLINEAR_REGRESSOR:
        REPRESENTATION_LAYER_INDEX = -6
        for i in range(4):
            x = layers.Dense(32, activation='relu')(x)

    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    print(y_train.shape)
    print(y_train_pseudo_class.shape)
    dataset = imbal.util.backend.MultiDatasetWithBatching(
        x_train,
        [y_train, y_train_pseudo_class],
        batch_size=BATCH_SIZE,
        shuffle=True,
        multi_output=True,
        mode=imbal.util.backend.constants.ModelType.REGRESSION
    )

    conflict_reduction_function = conflict_reduction_builder(
        conflict_reduction_method=CONFLICT_REDUCTION_METHOD,
        binning_method=GRADIENT_BINNING_METHOD,
        num_classes=NUM_CLASSES,
        normalize_before_conflict_reduction=NORMALIZE_GRADIENTS_BEFORE_REDUCTION
    )

    variables = model.trainable_variables

    shapes = [v.shape for v in variables]
    sizes = [tf.size(v) for v in variables]
    best_val_loss = None
    patience = 0
    best_loss_epoch = 0
    best_model_weights = model.get_weights()

    output_file = open(f"{OUTPUT_PATH}/log_{DATA_PREFIX}_{OUTPUT_POSTFIX}.txt", 'w')
    output_file.write(f"epoch,batch{''.join([f',bin_{x}_magnitude' for x in range(NUM_CLASSES)])}{''.join([f',original_to_bin_{x}_similarity' for x in range(NUM_CLASSES)])}{''.join([f',final_to_bin_{x}_similarity' for x in range(NUM_CLASSES)])}\n")

    for epoch in range(EPOCHS):
        total_loss = 0.0
        num_batches = 0

        for i in range(len(dataset)):
            (x_batch, y_batch, w_batch) = dataset[i]
            (y_reg_batch, y_pseudo_class_batch) = y_batch

            loss, original_gradient, final_gradient, binned_centroids = conflict_reduction_function(
                x_batch,
                y_reg_batch,
                y_pseudo_class_batch,
                model,
                loss_fn
            )

            split_gradients = tf.split(
                final_gradient,
                [int(size.numpy()) for size in sizes]
            )

            reconstructed_gradients = [
                tf.reshape(g, shape)
                for g, shape in zip(split_gradients, shapes)
            ]

            optimizer.apply_gradients(
                zip(reconstructed_gradients, variables)
            )

            print(f"Epoch {epoch + 1}/{EPOCHS} (Batch {i + 1}/{len(dataset)}), Loss: {loss.numpy():.6f}", end='\r')

            output_file.write(
                f"{epoch+1},"
                f"{i+1}"
                f"{''.join([f',{round(float(tf.norm(binned_centroids[x])), 5)}' for x in range(NUM_CLASSES)])}"
                f"{''.join([f',{round(float(cosine_similarity(original_gradient, binned_centroids[x])), 5)}' for x in range(NUM_CLASSES)])}"
                f"{''.join([f',{round(float(cosine_similarity(final_gradient, binned_centroids[x])), 5)}' for x in range(NUM_CLASSES)])}"
            )
            output_file.flush()

            if i < len(dataset) - 1:
                output_file.write(",-1\n")

            total_loss += loss.numpy()
            num_batches += 1
        dataset.on_epoch_end()

        avg_loss = total_loss / num_batches

        val_prediction = model(x_val, training=False)
        val_loss = loss_fn(y_val, val_prediction)

        patience += 1
        if best_val_loss is None or val_loss < best_val_loss - EARLY_STOPPING_DELTA:
            best_val_loss = val_loss
            best_model_weights = model.get_weights()
            best_loss_epoch = epoch + 1
            patience = 0


        output_file.write(f",{round(float(tf.reduce_mean(tf.abs(tf.reshape(val_prediction, [-1]) - tf.cast(tf.reshape(y_val, [-1]), tf.float32)))), 5)}\n")


        print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {avg_loss:.6f}, Validation Loss: {val_loss:.6f}" + " "*50)

        if patience > EARLY_STOPPING_PATIENCE:
            break
    output_file.close()
    print(f"Restoring weights from epoch {best_loss_epoch}...")
    model.set_weights(best_model_weights)
    print(f"Saving model...")
    model.save(f"{MODEL_OUTPUT_PATH}/{DATA_PREFIX}_{OUTPUT_POSTFIX}.keras")



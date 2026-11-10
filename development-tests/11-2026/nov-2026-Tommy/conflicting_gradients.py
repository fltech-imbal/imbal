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
    CONFLICT_REDUCTION_METHOD=middle_sum
):
    @tf.function(reduce_retracing=True)
    def get_sample_gradient(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            prediction = model(x[None, :], training=False)
            sample_loss = loss_fn(tf.reshape(y, [1, -1]), prediction)

        gradients = tape.gradient(sample_loss, model.trainable_variables)
        flattened_gradients = tf.concat(
            [
                tf.reshape(g, [-1])
                for g in gradients
                if g is not None
            ],
            axis=0
        )

        return gradients, flattened_gradients, sample_loss

    NUM_CLASSES = 3

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_sep_ec(
        f"{DATA_PATH}/{DATA_PREFIX}",
    )

    print(x_train[0].shape)

    y_train = np.digitize(y_train, bins=[-np.log(10), np.log(10)])
    y_test = np.digitize(y_test, bins=[-np.log(10), np.log(10)])
    y_val = np.digitize(y_val, bins=[-np.log(10), np.log(10)])

    y_train = tf.constant(y_train)
    y_test = tf.constant(y_test)
    y_val = tf.constant(y_val)

    x_train = tf.constant(x_train)
    y_train = tf.constant(y_train)
    x_val = tf.constant(x_val)
    y_val = tf.constant(y_val)
    x_test = tf.constant(x_test)
    y_test = tf.constant(y_test)

    f1 = tf.keras.metrics.F1Score(
        average='macro',
        threshold=None
    )

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

    outputs = layers.Dense(NUM_CLASSES)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    dataset = imbal.util.backend.DatasetWithBatching(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        mode=imbal.util.backend.constants.ModelType.CLASSIFICATION
    )

    variables = model.trainable_variables

    shapes = [v.shape for v in variables]
    sizes = [tf.size(v) for v in variables]
    best_val_loss = None
    patience = 0
    best_loss_epoch = 0
    best_model_weights = model.get_weights()

    output_file = open(f"{OUTPUT_PATH}/log_{DATA_PREFIX}_{OUTPUT_POSTFIX}.txt", 'w')
    output_file.write("epoch,batch,class_0_magnitude,class_1_magnitude,class_2_magnitude,original_to_class_0_similarity,original_to_class_1_similarity,original_to_class_2_similarity,final_to_class_0_similarity,final_to_class_1_similarity,final_to_class_2_similarity\n")

    for epoch in range(EPOCHS):
        total_loss = 0.0
        num_batches = 0

        last_normalized_original_gradient = None
        last_normalized_final_gradient = None
        last_class_centroids = [None] * NUM_CLASSES
        last_normalized_class_centroids = [None] * NUM_CLASSES

        for i in range(len(dataset)):
            (x_batch, y_batch, w_batch) = dataset[i]

            per_sample_gradients, per_sample_gradients_flattened, per_sample_loss = tf.vectorized_map(
                get_sample_gradient,
                (x_batch, y_batch),
            )

            loss = tf.reduce_mean(per_sample_loss)

            original_gradient = tf.reduce_sum(per_sample_gradients_flattened, axis=0)
            normalized_original_gradient = tf.math.l2_normalize(original_gradient, axis=0)
            last_normalized_original_gradient = normalized_original_gradient
            class_centroids, normalized_class_centroids = get_class_centroids(per_sample_gradients_flattened, y_batch, NUM_CLASSES)
            last_class_centroids = class_centroids
            last_normalized_class_centroids = normalized_class_centroids

            if CONFLICT_REDUCTION_METHOD is None:
                final_gradient = original_gradient
            else:
                conflict_reduced_class_gradients = middle_sum(class_centroids, normalized_class_centroids, NUM_CLASSES)
                sum_conflict_reduced = tf.reduce_sum(conflict_reduced_class_gradients, axis=0)
                final_gradient = sum_conflict_reduced / tf.norm(sum_conflict_reduced) * tf.norm(original_gradient, axis=0)

            last_normalized_final_gradient = tf.math.l2_normalize(final_gradient, axis=0)

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
                f"{i+1},"
                f"{round(float(tf.norm(last_class_centroids[0])), 5)},"
                f"{round(float(tf.norm(last_class_centroids[1])), 5)},"
                f"{round(float(tf.norm(last_class_centroids[2])), 5)},"
                f"{round(float(tf.reduce_sum(last_normalized_original_gradient * last_normalized_class_centroids[0])), 5)},"
                f"{round(float(tf.reduce_sum(last_normalized_original_gradient * last_normalized_class_centroids[1])), 5)},"
                f"{round(float(tf.reduce_sum(last_normalized_original_gradient * last_normalized_class_centroids[2])), 5)},"
                f"{round(float(tf.reduce_sum(last_normalized_final_gradient * last_normalized_class_centroids[0])), 5)},"
                f"{round(float(tf.reduce_sum(last_normalized_final_gradient * last_normalized_class_centroids[1])), 5)},"
                f"{round(float(tf.reduce_sum(last_normalized_final_gradient * last_normalized_class_centroids[2])), 5)},"
            )
            output_file.flush()

            if i < len(dataset) - 1:
                output_file.write("-1\n")

            total_loss += loss.numpy()
            num_batches += 1
        dataset.on_epoch_end()

        avg_loss = total_loss / num_batches

        test_prediction = model(x_val, training=False)
        val_loss = loss_fn(y_val, test_prediction)

        patience += 1
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            print(best_val_loss)
            best_model_weights = model.get_weights()
            best_loss_epoch = epoch + 1
            patience = 0

        f1.update_state(tf.one_hot(y_val, NUM_CLASSES), test_prediction)
        output_file.write(f"{round(float(f1.result().numpy()), 5)}\n")
        f1.reset_state()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {avg_loss:.6f}, Validation Loss: {val_loss:.6f}" + " "*50)
        for i in range(NUM_CLASSES):
            print(f"Magnitude of gradient for class {i}: {round(float(tf.norm(last_class_centroids[i])), 5)}")
        for i in range(NUM_CLASSES):
            print(f"Similarity of original gradient to class {i}: {round(float(tf.reduce_sum(last_normalized_original_gradient * last_normalized_class_centroids[i])), 5)}")
        if CONFLICT_REDUCTION_METHOD is not None:
            for i in range(NUM_CLASSES):
                print( f"Similarity of final gradient to class {i}: {round(float(tf.reduce_sum(last_normalized_final_gradient * last_normalized_class_centroids[i])), 5)}")

        if patience > EARLY_STOPPING_PATIENCE:
            break
    output_file.close()
    print(f"Restoring weights from epoch {best_loss_epoch}...")
    model.set_weights(best_model_weights)
    print(f"Saving model...")
    model.save(f"{MODEL_OUTPUT_PATH}/{DATA_PREFIX}_{OUTPUT_POSTFIX}.keras")



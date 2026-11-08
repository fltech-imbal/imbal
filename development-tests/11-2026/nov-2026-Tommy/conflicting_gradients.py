import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import sys
import imbal


@tf.function(reduce_retracing=True)
def normalize(vec):
    if tf.norm(vec) == 0:
        return tf.constant([0.0, 0.0])
    return vec / tf.norm(vec)

@tf.function(reduce_retracing=True)
def cosine_similarity(a, b):
    denom = tf.norm(a) * tf.norm(b)
    if denom < 10e-6:
        return tf.constant(0.0)
    return tf.reduce_sum(a * b) / denom

@tf.function(reduce_retracing=True)
def project(vec_1, vec_2):
    if tf.norm(vec_2) < 10e-6:
        return tf.constant([0.0, 0.0])
    return tf.reduce_sum(vec_2 * vec_1) / tf.reduce_sum(vec_2 * vec_2) * vec_2

@tf.function(reduce_retracing=True)
def get_class_centroids(gradients, labels, num_classes):
    centroids = []
    for i in range(num_classes):
        mask = tf.reshape(labels == i, [-1])
        class_per_sample_gradients = tf.boolean_mask(gradients, mask)
        centroid = tf.reduce_sum(class_per_sample_gradients, axis=0)
        centroids.append(centroid)

    normalized_centroids = [tf.math.l2_normalize(c, axis=0) for c in centroids]
    return centroids, normalized_centroids


@tf.function(reduce_retracing=True)
def get_conflict_reduced(class_gradients, normalized_class_gradients, num_classes):
    return [gi_minus_proj_sum_cos_times_gj(normalized_class_gradients, i) for i in range(num_classes)]


@tf.function(reduce_retracing=True)
def gi_minus_proj_sum_cos_times_gj(vectors, index):
    current = vectors[index]
    sum_conflicting = tf.zeros_like(current)
    for i in range(len(vectors)):
        if i == index: continue
        other = vectors[i]
        similarity = cosine_similarity(current, other)
        if similarity < 0:
            sum_conflicting += other * similarity
    projected_vector = project(current, sum_conflicting)
    return current - projected_vector


@tf.function(reduce_retracing=True)
def projection(a, b):
    numerator = tf.reduce_sum(tf.multiply(a, b))
    denominator = tf.reduce_sum(tf.multiply(b, b))
    return b * numerator / denominator

def run_model(
    LEARNING_RATE = 5e-5,
    WEIGHT_CANDIDATES = [0.1, 0.3, 0.5, 0.7],
    SINGLE_WEIGHT_ALPHA = 1,

    DATA_PATH = "cleaned-dtw-SEP-EC-data",
    DATA_PREFIX = 'sep_e_log_normalized',
    MODEL_OUTPUT_PATH = "models-sep-e",
    UNIT_REPRESENTATIONS = False,
    NONLINEAR_REGRESSOR = False,
    LEARNABLE_RATIO = False,
    OUTPUT_POSTFIX = '_tempdelete',

    REPRESENTATION_LAYER_INDEX = -2,
    EARLY_STOPPING_PATIENCE = 100,
    EPOCHS = 10000,

    STRATIFY = True,
    BATCH_SIZE = 2048,
    KDE_BIN_COUNT=64,

    CONFLICT_REDUCTION_METHOD= get_conflict_reduced
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

    def load_sep_ec(path_prefix):
        training_data = pd.read_csv(path_prefix + '_training.csv')
        test_data = pd.read_csv(path_prefix + '_test.csv')
        val_data = pd.read_csv(path_prefix + '_validation.csv')
        training_labels = training_data.pop("p16.4_tplus6")
        val_labels = val_data.pop("p16.4_tplus6")
        test_labels = test_data.pop("p16.4_tplus6")
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

    print(x_train[0].shape)

    y_train = np.digitize(y_train, bins=[-np.log(10), np.log(10)])
    y_test = np.digitize(y_test, bins=[-np.log(10), np.log(10)])
    y_val = np.digitize(y_val, bins=[-np.log(10), np.log(10)])

    y_train = tf.constant(y_train)
    y_test = tf.constant(y_test)
    y_val = tf.constant(y_val)

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

    # """
    # Generate sample weights
    # """
    #
    # sample_weights = imbal.classification.generate_sample_weights(
    #
    # )

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
    best_test_loss = None

    for epoch in range(EPOCHS):
        total_loss = 0.0
        num_batches = 0

        last_normalized_original_gradient = None
        last_normalized_final_gradient = None
        last_normalized_class_centroids = [None] * NUM_CLASSES
        patience = 0

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
            last_normalized_class_centroids = normalized_class_centroids

            if CONFLICT_REDUCTION_METHOD is None:
                final_gradient = original_gradient
            else:
                conflict_reduced_class_gradients = get_conflict_reduced(class_centroids, normalized_class_centroids, NUM_CLASSES)
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

            total_loss += loss.numpy()
            num_batches += 1
        dataset.on_epoch_end()

        avg_loss = total_loss / num_batches

        test_prediction = model(x_test, training=False)
        test_loss = loss_fn(y_test, test_prediction)

        patience += 1
        if best_test_loss is None or test_loss < best_test_loss:
            best_test_loss = test_loss
            print(best_test_loss)
            patience = 0

        print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {avg_loss:.6f}, Test Loss: {test_loss:.6f}" + " "*50)
        if CONFLICT_REDUCTION_METHOD is not None:
            for i in range(NUM_CLASSES):
                print(f"Similarity of original gradient to class {i}: {round(float(tf.reduce_sum(last_normalized_original_gradient * last_normalized_class_centroids[i])), 5)}")
            for i in range(NUM_CLASSES):
                print( f"Similarity of final gradient to class {i}: {round(float(tf.reduce_sum(last_normalized_final_gradient * last_normalized_class_centroids[i])), 5)}")

        if patience > EARLY_STOPPING_PATIENCE:
            break

    model.save(f"{MODEL_OUTPUT_PATH}/{DATA_PREFIX}_{OUTPUT_POSTFIX}.keras")

run_model(
    EPOCHS=200,
    EARLY_STOPPING_PATIENCE=0,
    OUTPUT_POSTFIX='test_1'
)
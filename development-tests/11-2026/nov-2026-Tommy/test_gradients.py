import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import sys
import imbal

DATA_PATH = "cleaned-dtw-SEP-EC-data"
DATA_PREFIX = 'sep_e_log_normalized'
UNIT_REPRESENTATIONS = False
NONLINEAR_REGRESSOR = False
LEARNING_RATE = 2e-4

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

y_train = (y_train >= np.log(10)).astype(int)
y_test = (y_test >= np.log(10)).astype(int)
y_val = (y_val >= np.log(10)).astype(int)

# LAYER_DIMS = [128, 128, 128, 64, 64, 64, 32, 32, 32]
LAYER_DIMS = [128, 64, 64, 32, 32]

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

loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

epochs = 100
batch_size = 2048

dataset = imbal.util.backend.DatasetWithBatching(
    x_train,
    y_train,
    batch_size=batch_size,
    shuffle=True,
    mode=imbal.util.backend.constants.ModelType.REGRESSION
)

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

@tf.function(reduce_retracing=True)
def projection(a, b):
    numerator = tf.reduce_sum(tf.multiply(a, b))
    denominator = tf.reduce_sum(tf.multiply(b, b))
    return b * numerator / denominator

variables = model.trainable_variables

shapes = [v.shape for v in variables]
sizes = [tf.size(v) for v in variables]

for epoch in range(epochs):
    total_loss = 0.0
    num_batches = 0
    last_pre_reduction_similarity = None
    last_post_reduction_similarity = None
    pos_final_similarity = None
    neg_final_similarity = None

    for i in range(len(dataset)):
        (x_batch, y_batch, w_batch) = dataset[i]

        per_sample_gradients, per_sample_gradients_flattened, per_sample_loss = tf.vectorized_map(
            get_sample_gradient,
            (x_batch, y_batch),
        )

        loss = tf.reduce_mean(per_sample_loss)

        negative_mask = tf.reshape(y_batch == 0, [-1])
        negative_per_sample_gradients = tf.boolean_mask(per_sample_gradients_flattened, negative_mask)

        positive_mask = tf.reshape(y_batch == 1, [-1])
        positive_per_sample_gradients = tf.boolean_mask(per_sample_gradients_flattened, positive_mask)

        negative_gradient_centroid = tf.reduce_sum(negative_per_sample_gradients, axis=0)
        positive_gradient_centroid = tf.reduce_sum(positive_per_sample_gradients, axis=0)

        normalized_negative_centroid = tf.math.l2_normalize(negative_gradient_centroid, axis=0)
        normalized_positive_centroid = tf.math.l2_normalize(positive_gradient_centroid, axis=0)
        original_normalized_negative_centroid = normalized_negative_centroid
        original_normalized_positive_centroid = normalized_positive_centroid

        centroid_similarity = tf.reduce_sum(normalized_positive_centroid * normalized_negative_centroid)
        last_pre_reduction_similarity = float(centroid_similarity)

        if (centroid_similarity < 0):
            projected_negative_gradient_centroid = negative_gradient_centroid - projection(negative_gradient_centroid, positive_gradient_centroid)
            projected_positive_gradient_centroid = positive_gradient_centroid - projection(positive_gradient_centroid, negative_gradient_centroid)

            negative_gradient_centroid = projected_negative_gradient_centroid
            positive_gradient_centroid = projected_positive_gradient_centroid

            normalized_negative_centroid = tf.math.l2_normalize(negative_gradient_centroid, axis=0)
            normalized_positive_centroid = tf.math.l2_normalize(positive_gradient_centroid, axis=0)

        centroid_similarity = tf.reduce_sum(normalized_positive_centroid * normalized_negative_centroid)
        last_post_reduction_similarity = float(centroid_similarity)

        flat_sum_gradients = tf.reduce_sum([negative_gradient_centroid, positive_gradient_centroid], axis=0)
        final_gradient = (normalized_negative_centroid + normalized_positive_centroid) / 2 * tf.linalg.norm(flat_sum_gradients, axis=0)

        final_gradient_normalized = tf.math.l2_normalize(final_gradient, axis=0)

        neg_final_similarity = float(tf.reduce_sum(final_gradient_normalized * original_normalized_negative_centroid))
        pos_final_similarity = float(tf.reduce_sum(final_gradient_normalized * original_normalized_positive_centroid))

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

        last_gradients = per_sample_gradients

        print(f"Epoch {epoch + 1}/{epochs} (Batch {i + 1}/{len(dataset)}), Loss: {loss.numpy():.6f}", end='\r')

        total_loss += loss.numpy()
        num_batches += 1
    dataset.on_epoch_end()

    avg_loss = total_loss / num_batches

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}" + " "*50)
    print(f"Similarity of negative-positive sample centroids before projection: {round(last_pre_reduction_similarity, 5)}")
    print(f"Similarity of negative-positive sample centroids after projection: {round(last_post_reduction_similarity, 5)}")
    print(f"Similarity of negative sample centroid with resulting gradient: {round(neg_final_similarity, 5)}")
    print(f"Similarity of positive sample centroid with resulting gradient: {round(pos_final_similarity, 5)}")


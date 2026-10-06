import matplotlib.pyplot as plt
import numpy as np
from imbal import util
import keras
import tensorflow as tf

def plot_similarity(
    model,
    x,
    y,
    representation_layer_index=-2,
    unit_representation=False,
    save_figure=None,
    cmap='viridis',
    s=10
):
    x = np.array(x)
    y = np.array(y)
    y = y.reshape(-1)

    if representation_layer_index < 0:
        representation_layer_index = len(model.layers) + representation_layer_index

    found_layer, found_index = util.get_representation_layer_index(
        model,
        desired_layer_index=representation_layer_index
    )

    intermediate_model = keras.Model(inputs=model.input,
                                     outputs=found_layer.output)

    latents = intermediate_model.predict(x)

    if unit_representation:
        x_normalized = tf.math.l2_normalize(x, axis=1)  # [N, D]

        similarity = tf.matmul(
            x_normalized,
            x_normalized,
            transpose_b=True
        )
    else:
        squared_norms = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)  # [N, 1]

        squared_distances = (
                squared_norms
                + tf.transpose(squared_norms)
                - 2.0 * tf.matmul(x, x, transpose_b=True)
        )

        squared_distances = tf.maximum(squared_distances, 0.0)

        similarity = -tf.sqrt(squared_distances)

    similarity = np.array(similarity).reshape(-1)

    label_1, label_2 = tf.meshgrid(y, y)
    label_pairs = tf.stack([label_1, label_2], axis=-1)

    x_coords = np.array(label_pairs[:, :, 0]).reshape(-1)
    y_coords = np.array(label_pairs[:, :, 1]).reshape(-1)

    sort_indices = tf.argsort(similarity)
    similarity = similarity[sort_indices]
    x_coords = x_coords[sort_indices]
    y_coords = y_coords[sort_indices]

    plt.figure(figsize=(7, 6))
    plt.scatter(x_coords, y_coords, c=similarity, cmap=cmap, s=s)
    plt.colorbar()

    if save_figure is not None:
        plt.savefig(save_figure)
    # plt.show()
    plt.cla()

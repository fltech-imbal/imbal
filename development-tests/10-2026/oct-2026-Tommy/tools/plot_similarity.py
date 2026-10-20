import matplotlib.pyplot as plt
import numpy as np
from imbal import util
import keras
import tensorflow as tf
from math import floor, ceil

# def plot_similarity(
#     model,
#     x,
#     y,
#     representation_layer_index=-2,
#     unit_representation=False,
#     save_figure=None,
#     cmap='jet',
#     s=10
# ):
#     x = np.array(x)
#     y = np.array(y)
#     y = y.reshape(-1)
#     sorted_indices = np.argsort(y)
#     x = x[sorted_indices]
#     y = y[sorted_indices]
#
#     if representation_layer_index < 0:
#         representation_layer_index = len(model.layers) + representation_layer_index
#
#     found_layer, found_index = util.get_representation_layer_index(
#         model,
#         desired_layer_index=representation_layer_index
#     )
#
#     intermediate_model = keras.Model(inputs=model.input,
#                                      outputs=found_layer.output)
#
#     reps = intermediate_model.predict(x)
#
#     if unit_representation:
#         reps_normalized = tf.math.l2_normalize(reps, axis=1)  # [N, D]
#
#         similarity = tf.matmul(
#             reps_normalized,
#             reps_normalized,
#             transpose_b=True
#         )
#     else:
#         squared_norms = tf.reduce_sum(tf.square(reps), axis=1, keepdims=True)  # [N, 1]
#
#         squared_distances = (
#                 squared_norms
#                 + tf.transpose(squared_norms)
#                 - 2.0 * tf.matmul(reps, reps, transpose_b=True)
#         )
#
#         squared_distances = tf.maximum(squared_distances, 0.0)
#
#         similarity = -tf.sqrt(squared_distances)
#
#     similarity = np.array(similarity).reshape(-1)
#
#     label_1, label_2 = tf.meshgrid(y, y)
#     label_pairs = tf.stack([label_1, label_2], axis=-1)
#
#     x_coords = np.array(label_pairs[:, :, 0]).reshape(-1)
#     y_coords = np.array(label_pairs[:, :, 1]).reshape(-1)
#
#     sort_indices = tf.argsort(similarity)
#     similarity = similarity[sort_indices]
#     x_coords = x_coords[sort_indices]
#     y_coords = y_coords[sort_indices]
#
#     plt.figure(figsize=(7, 6))
#     plt.scatter(x_coords, y_coords, c=similarity, cmap=cmap, s=s)
#     plt.colorbar()
#
#     if save_figure is not None:
#         plt.savefig(save_figure)
#     # plt.show()
#     plt.cla()

def plot_similarity_binned(
    model,
    x,
    y,
    representation_layer_index=-2,
    unit_representation=False,
    save_figure=None,
    cmap='jet',
    bins=64,
    ticks=10,
    rank_based=False
):
    x = np.array(x)
    y = np.array(y)
    y = y.reshape(-1)
    sorted_indices = np.argsort(y)
    x = x[sorted_indices]
    y = y[sorted_indices]

    if representation_layer_index < 0:
        representation_layer_index = len(model.layers) + representation_layer_index

    found_layer, found_index = util.get_representation_layer_index(
        model,
        desired_layer_index=representation_layer_index
    )

    intermediate_model = keras.Model(inputs=model.input,
                                     outputs=found_layer.output)

    reps = intermediate_model.predict(x)

    if unit_representation:
        reps_normalized = tf.math.l2_normalize(reps, axis=1)  # [N, D]

        similarity = tf.matmul(
            reps_normalized,
            reps_normalized,
            transpose_b=True
        )
        plt.title("Similarity Matrix")
    else:
        squared_norms = tf.reduce_sum(tf.square(reps), axis=1, keepdims=True)  # [N, 1]

        squared_distances = (
                squared_norms
                + tf.transpose(squared_norms)
                - 2.0 * tf.matmul(reps, reps, transpose_b=True)
        )

        squared_distances = tf.maximum(squared_distances, 0.0)

        similarity = tf.sqrt(squared_distances)
        plt.title('Distance Matrix')

    matrix = np.zeros((bins, bins))

    label_min = np.min(y)
    label_max = np.max(y)
    label_range = label_max - label_min
    label_frac = label_range / bins
    num_values = len(y)

    if rank_based:
        edges = np.linspace(0, num_values, bins + 1, dtype=int)

        for i in range(bins):
            for j in range(bins):
                bin_similarities = similarity[
                                   edges[i]:edges[i + 1],
                                   edges[j]:edges[j + 1]
                                   ]
                matrix[i, j] = np.mean(bin_similarities)
    else:
        for i in range(bins):
            for j in range(bins):
                i_comparison = (y >= label_min + label_frac*i) & (y < label_min + label_frac*(i+1))
                j_comparison = (y >= label_min + label_frac*j) & (y < label_min + label_frac*(j+1))
                bin_similarities = tf.boolean_mask(similarity, i_comparison, axis=0)
                bin_similarities = tf.boolean_mask(bin_similarities, j_comparison, axis=1)
                matrix[i, j] = np.mean(bin_similarities)

    plt.imshow(matrix, origin='lower', interpolation='nearest', cmap=cmap)
    ax = plt.gca()

    ax.set_xticks([i*bins/(ticks-1) for i in range(ticks)])
    ax.set_yticks([i * bins / (ticks - 1) for i in range(ticks)])
    if rank_based:
        ax.set_xticklabels([round(i*num_values/ticks) for i in range(ticks)])
        ax.set_yticklabels([round(i*num_values/ticks) for i in range(ticks)])
    else:
        ax.set_xticklabels([round(label_min + label_range / (ticks-1) * i, 2) for i in range(ticks)])
        ax.set_yticklabels([round(label_min + label_range / (ticks-1) * i, 2) for i in range(ticks)])
    plt.colorbar()

    if save_figure is not None:
        plt.savefig(save_figure)
    plt.show()


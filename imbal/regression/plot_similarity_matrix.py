import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from imbal import util


def plot_similarity_matrix(
    model,
    x,
    y,
    representation_layer_index=-2,
    unit_representation=False,
    cmap='jet',
    bins=64,
    ticks=10,
    save_figure=None
):
    """
    Plots and displays a comparison of the true labels and the predicted labels
    for regression data. Points plotted are (true, prediction) for each label/prediction
    pair, such that a perfect prediction would fall precisely on the line :math:`y=x`.
    Prediction-label pairs will be plotted in the same order as supplied to the function.

    Args:
        model: The TensorFlow model whose representation space you wish to visualize.
        x: The data you wish to generate a similarity matrix for.
        y: The corresponding labels for the provided data, as a column vector. These labels are
            used to sort and bin the provided data.
        representation_layer_index: Optional, default :math:`-2`. The index of the layer of your
            model to extract the representation from. Defaults to the second to last
            layer of the provided model.
        unit_representation: Optional, default :code:`False`. Whether the provided model
            uses a unit hypersphere representation. If set to :code:`True`, cosine similarity
            will be used to compute the similarity between representations as opposed to distance.
        cmap: Optional, default :code:`'jet'`. The matplotlib colormap to use. See
            `matplotlib's documentation <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_
            for a list of available colormaps.
        bins: Optional, default :code:`64`. The number of bins to use across the label range. The lower
            the amount of bins, the more coarse the resulting matrix will be, but the faster it will be
            generated. A higher amount of bins produces a more fine-grain matrix, but takes longer to
            generate.
        ticks: Optional, default :code:`10`. The number of label ticks to plots along each
            axis of the generated matrix.
        save_figure: Optional, default :code:`None`. If set to a string, will save the
            resultant plot to the specified path.

    Example:

    .. code::

        from imbal.regression import plot_similarity_matrix
        import numpy as np

        model = tf.keras.models.load_model('model.keras')

        x = np.random.randn(100, 100) # or some other data
        y = np.random.randn(100, 1) # or some other labels

        # Plots the similarity matrix and saves it to "similarity_matrix.png"
        plot_similarity_matrix(
            model,
            x,
            y,
            save_figure="similarity_matrix.png"
        )

    The code above will produce a plot similar to the following:

    .. image::
       :alt: A similarity matrix plot
       :width: 400px
       :align: center

    """

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
    ax.set_xticklabels([round(label_min + label_range / (ticks-1) * i, 2) for i in range(ticks)])
    ax.set_yticklabels([round(label_min + label_range / (ticks-1) * i, 2) for i in range(ticks)])
    plt.colorbar()

    if save_figure is not None:
        plt.savefig(save_figure)
    plt.show()


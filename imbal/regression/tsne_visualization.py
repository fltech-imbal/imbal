import numpy as np
from imbal.util.visualization import generate_tsne_visualization

def tsne_visualization(
    model,
    data,
    labels,
    latent_layer_index=-2,
    gradient='plasma',
    perplexity=30,
    save_figure=None,
    s=None,
    marker=None,
    bin_count=64,
    padding_factor=0.01,
):
    """
    Args:
        model: The PyTorch model whose representation space you wish to visualize.
        data: The data whose representation you wish to visualize, as a column vector.
        labels: The corresponding labels for the provided data, as a column vector.
        latent_layer_index: Optional, default :math:`2`. The index of the layer of your
            model to extract the representation from. Defaults to the second to last
            layer of the provided model.
        perplexity: Optional, default :math:`30`. See `sklearn.manifold.TSNE <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html>`_.
        save_figure: Optional, default :code:`None`. If set to a string, will save the
            resultant plot to the specified path.
        s: Optional, default :code:`None`. If not :code:`None`, a float that represents the marker size
            of each plotted point. See `matplotlib.pyplot.scatter <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html>`_.
        gradient: Optional, default :code:`'plasma'`. A gradient to be used for plotting. See `matplotlib's colormaps <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_
        marker: Optional, default :code:`None`. If not :code:`None`, a marker shape. See `matplotlib.pyplot.scatter <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html>`_.
        bin_count: Optional, default :code:`64`. The number of bins to use for determining graphing order. Labels
            are grouped into bins, and then bins are plotted from most frequent to least frequenct, ensuring
            that rare points are always visible "on top" of more common points.
        padding_factor: Optional, default :code:`0.01`. Used to add a small padding to
            the data range used for binning for the histogram. See :doc:`imbal.regression.fit_kde </imbal/regression/fit_kde>`.

    Returns: :code:`None`

    Example:

    .. code-block:: python

        >>> # For this example, assume a trained model is saved in 'model', and
        >>> # data and labels are stored in 'data' and 'labels' respectively.

        >>> imbal.regression.tsne_visualization(
        >>>     model,
        >>>     data,
        >>>     labels,
        >>>     perplexity=20
        >>> )

    """
    indices = np.argsort(labels)

    sorted_labels = labels[indices]
    sorted_data = data[indices]

    fig = generate_tsne_visualization(
        model,
        sorted_data,
        sorted_labels,
        latent_layer_index=latent_layer_index,
        gradient=gradient,
        save_figure=save_figure,
        perplexity=perplexity,
        s=s,
        marker=marker,
        bin_count=bin_count,
        padding_factor=padding_factor,
        mode='regression',
    )
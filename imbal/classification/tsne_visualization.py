from imbal.util.backend.visualization import generate_tsne_visualization

def tsne_visualization(
    model,
    data,
    labels,
    representation_layer_index=-2,
    perplexity=30,
    s=None,
    c=None,
    marker=None,
    save_figure=None
):
    """

    Provides a visualization of a model's representation space using
    TSNE.

    This function will always plot rarer classes last, allowing for those rare
    classes to always be more visible in the TSNE plot. This is because points
    plotted first may be overwritten by points plotted later, so plotting rare
    classes last ensures the rare classes are not overwritten by common classes.
    An example of this can be seen below.

    Args:
        model: The TensorFlow model whose representation space you wish to visualize.
        data: The data whose representation you wish to visualize, as a column vector.
        labels: The corresponding labels for the provided data, as a column vector.
        representation_layer_index: Optional, default :math:`-2`. The index of the layer of your
            model to extract the representation from. Defaults to the second to last
            layer of the provided model.
        perplexity: Optional, default :math:`30`. See `sklearn.manifold.TSNE <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html>`_.
            The suggested perplexity value from the `paper which introduced t-SNE <https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf>`_
            is from 5 to 50.
        s: Optional, default :code:`None`. If not :code:`None`, a list of floats of length
            equal to the number of classes, where each float represents the marker size for the
            class when plotted, in sorted order. See `matplotlib.pyplot.scatter <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html>`_.
        c: Optional, default :code:`None`. If not :code:`None`, a list of colors for each
            class when plotted, in sorted order. See `matplotlib.pyplot.scatter <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html>`_.
        marker: Optional, default :code:`None`. If not :code:`None`, a list of marker shapes for each
            class when plotted, in sorted order. See `matplotlib.pyplot.scatter <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html>`_.
        save_figure: Optional, default :code:`None`. If set to a string, the
            resultant plot with be saved to the specified path.

    Returns: :code:`None`

    Example:

    .. code-block:: python

        >>> # For this example, assume a trained model is saved in 'model', and
        >>> # data and labels are stored in 'data' and 'labels' respectively.

        >>> imbal.classification.tsne_visualization(
        >>>     model,
        >>>     data,
        >>>     labels,
        >>>     perplexity=20
        >>> )

    """

    fig = generate_tsne_visualization(
        model,
        data,
        labels,
        representation_layer_index=representation_layer_index,
        save_figure=save_figure,
        perplexity=perplexity,
        mode='classification',
        s=s,
        c=c,
        marker=marker
    )
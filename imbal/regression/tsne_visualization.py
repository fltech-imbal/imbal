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
        padding_factor:
        bin_count:
        marker:
        s:
        model:
        data:
        labels:
        latent_layer_index:
        gradient:
        perplexity:
        save_figure:

    Returns:

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
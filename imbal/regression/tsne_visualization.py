import numpy as np
from imbal.util.visualization import generate_tsne_visualization


def tsne_visualization(
    model,
    data,
    labels,
    sort='ascending',
    latent_layer_index=-2,
    gradient='plasma',
    save_figure=None
):
    indices = np.argsort(data)

    if sort == 'descending':
        indices = indices[::-1]
    sorted_labels = labels[indices]
    sorted_data = data[indices]

    fig = generate_tsne_visualization(
        model,
        sorted_data,
        sorted_labels,
        latent_layer_index=latent_layer_index,
        gradient=gradient,
        save_figure=save_figure
    )
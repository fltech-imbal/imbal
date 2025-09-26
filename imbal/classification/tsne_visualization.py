import numpy as np
from imbal.util.visualization import generate_tsne_visualization

def tsne_visualization(
    model,
    data,
    labels,
    latent_layer_index=-2,
    gradient='plasma',
    perplexity=30,
    save_figure=None
):
    unique_classes, counts = np.unique(labels, return_counts=True)
    indices = np.argsort(counts)[::-1]
    unique_classes = unique_classes[indices]

    sorted_labels = np.concatenate([labels[labels == cls] for cls in unique_classes])
    sorted_data = np.concatenate([data[labels == cls] for cls in unique_classes])

    fig = generate_tsne_visualization(
        model,
        sorted_data,
        sorted_labels,
        latent_layer_index=latent_layer_index,
        gradient=gradient,
        save_figure=save_figure,
        perplexity=perplexity
    )
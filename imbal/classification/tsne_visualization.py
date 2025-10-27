import numpy as np
from imbal.util.visualization import generate_tsne_visualization

def tsne_visualization(
    model,
    data,
    labels,
    latent_layer_index=-2,
    display_classes=None,
    gradient='plasma',
    perplexity=30,
    save_figure=None
):
    if display_classes is None:
        display_classes = np.array([
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf"
        ])

    unique_classes, counts = np.unique(labels, return_counts=True)
    indices = np.argsort(counts)[::-1]
    ordered_unique_classes = unique_classes[indices]

    sorted_labels = np.concatenate([labels[labels == cls] for cls in ordered_unique_classes])
    sorted_data = np.concatenate([data[labels == cls] for cls in ordered_unique_classes])
    sorted_colors = display_classes[np.searchsorted(unique_classes, sorted_labels)]


    fig = generate_tsne_visualization(
        model,
        sorted_data,
        sorted_labels,
        latent_layer_index=latent_layer_index,
        gradient=gradient,
        save_figure=save_figure,
        perplexity=perplexity,
        legend_pairs=(unique_classes, display_classes),
        color_map=sorted_colors
    )
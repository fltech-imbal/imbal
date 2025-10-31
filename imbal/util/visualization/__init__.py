import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow import keras
import numpy as np
from imbal.util.sample_weighting import get_label_bin_bounds

def generate_tsne_visualization(
        model,
        data,
        labels,
        latent_layer_index,
        gradient=None,
        mode='classification',
        save_figure=None,
        perplexity=30,
        bin_count=64,
        padding_factor=0.01,
        s=None,
        c=None,
        marker=None
):
    intermediate_model = keras.Model(inputs=model.input,
                                     outputs=model.get_layer(index=latent_layer_index).output)

    latents = intermediate_model.predict(data)

    tsne = TSNE(n_components=2, random_state=None, perplexity=perplexity)
    tsne_fit = tsne.fit_transform(latents)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    scatter = None
    if mode == 'classification':
        unique_classes, counts = np.unique(labels, return_counts=True)
        index_ordering = np.argsort(counts)[::-1]
        unique_classes = unique_classes[index_ordering]

        if s is not None:
            s = np.array(s)[index_ordering]
        if c is not None:
            c = np.array(c)[index_ordering]
        if marker is not None:
            marker = np.array(marker)[index_ordering]
        for i in range(len(unique_classes)):
            cls_s = s[i] if s is not None else None
            cls_c = c[i] if c is not None else None
            cls_marker = marker[i] if marker is not None else None
            scatter = ax.scatter(
                tsne_fit[:, 0][labels == unique_classes[i]],
                tsne_fit[:, 1][labels == unique_classes[i]],
                label=unique_classes[i],
                s=cls_s,
                c=cls_c,
                marker=cls_marker
            )
        ax.legend()
    else:
        label_min, label_max, step = get_label_bin_bounds(labels, bin_count, padding_factor)

        bins = [np.where((labels >= label_min[0] + step * i) & (labels < label_min[0] + step * (i + 1)))[0] for i in range(bin_count)]
        sorted_bins = sorted(bins, key=len, reverse=True)

        for indices in sorted_bins:
            scatter = ax.scatter(
                tsne_fit[:, 0][indices],
                tsne_fit[:, 1][indices],
                cmap=gradient,
                c=labels[indices],
                vmin=label_min[0],
                vmax=label_max[0],
                s=s,
                marker=marker
            )
        plt.colorbar(scatter)

    assert scatter is not None

    if save_figure is not None:
        plt.savefig(save_figure)

    plt.show()

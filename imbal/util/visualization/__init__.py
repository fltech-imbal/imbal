import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.manifold import TSNE
from tensorflow import keras

def generate_tsne_visualization(
        model,
        data,
        labels,
        latent_layer_index,
        gradient,
        save_figure=None,
        perplexity=30,
        color_map=None,
        legend_pairs=None,
):
    intermediate_model = keras.Model(inputs=model.input,
                                     outputs=model.get_layer(index=latent_layer_index).output)

    latents = intermediate_model.predict(data)

    tsne = TSNE(n_components=2, random_state=None, perplexity=perplexity)
    tsne_fit = tsne.fit_transform(latents)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    if color_map is None:
        scatter = ax.scatter(tsne_fit[:, 0], tsne_fit[:, 1], c=labels, cmap=gradient)
        plt.colorbar(scatter)
    else:
        handles = [patches.Patch(color=legend_pairs[1][i], label=legend_pairs[0][i]) for i in range(legend_pairs[0].shape[0])]
        scatter = ax.scatter(tsne_fit[:, 0], tsne_fit[:, 1], c=color_map)
        plt.legend(handles=handles)

    if save_figure is not None:
        plt.savefig(save_figure)

    plt.show()

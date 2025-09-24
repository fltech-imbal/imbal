import os

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow import keras

def generate_tsne_visualization(
        model,
        data,
        labels,
        latent_layer_index,
        gradient,
        save_figure=None
):
    intermediate_model = keras.Model(inputs=model.input,
                                     outputs=model.get_layer(index=latent_layer_index).output)
    latents = intermediate_model.predict(data)

    tsne = TSNE(n_components=2, random_state=0)
    tsne_fit = tsne.fit_transform(latents)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(tsne_fit[:, 0], tsne_fit[:, 1], c=labels, cmap=gradient)
    plt.colorbar(scatter)

    if save_figure is not None:
        print('HELLLOOOO', save_figure)
        print(os.getcwd())
        plt.savefig(save_figure)

    plt.show()

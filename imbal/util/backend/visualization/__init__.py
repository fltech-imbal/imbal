import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow import keras
import numpy as np
from imbal.util.backend.sample_weighting import get_label_bin_bounds
from imbal import util
import tensorflow as tf

def generate_tsne_visualization(
        model,
        data,
        labels,
        representation_layer_index=-2,
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

    if representation_layer_index < 0:
        representation_layer_index =  len(model.layers) + representation_layer_index

    found_layer, found_index = util.get_representation_layer_index(
        model,
        desired_layer_index=representation_layer_index
    )

    intermediate_model = keras.Model(inputs=model.input,
                                     outputs=found_layer.output)

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

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

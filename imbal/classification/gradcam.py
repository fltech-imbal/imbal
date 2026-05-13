import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import matplotlib as mpl

def _find_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError(
        "Could not automatically find a Conv2D layer. "
        "Please pass last_conv_layer_name explicitly."
    )


def gradcam_explain_image_sample(
    sample,
    model,
    last_conv_layer_name=None,
    class_names=None,
    label_to_explain=None,
    actual_label=None,
    preprocess_fn=None,
    alpha=0.4,
    importance_threshold=0.5,
    cmap="jet",
    figure_save_path="gradcam-explanation.png",
    save_figure=False,
    show=True,
    return_heatmap=False,
):
    """
    Generates a Grad-CAM explanation for a single image classification sample.

    Args:
        sample: A single image as a NumPy array, shaped (H, W, C) or (1, H, W, C).
        model: A Keras-compatible image classification model.
        last_conv_layer_name: Optional name of the convolutional layer to explain.
            If None, the last Conv2D layer is used.
        label_to_explain: Optional class/output index to explain. If None, the
            predicted class is used. For binary sigmoid outputs, index 0 is used.
        class_names: Optional list of class names.
        actual_label: Optional true class index, displayed in the plot title.
        preprocess_fn: Optional preprocessing function applied before prediction.
        alpha: Maximum heatmap overlay opacity.
        importance_threshold: Minimum normalized importance required for a
            region to appear in the overlay.
        cmap: Matplotlib colormap name.
        figure_save_path: Path where the figure is saved if save_figure=True.
        save_figure: Whether to save the visualization.
        show: Whether to display the visualization.
        return_heatmap: Whether to return the raw heatmap too.

    Returns:
    If return_heatmap is False:
        numpy.ndarray:
            Superimposed RGB Grad-CAM visualization image with
            shape (H, W, 3) and dtype uint8.

    If return_heatmap is True:
        tuple[numpy.ndarray, numpy.ndarray]:
            A tuple containing:

            - superimposed_img:
              RGB Grad-CAM visualization image with shape (H, W, 3)
              and dtype uint8.

            - heatmap:
              Normalized Grad-CAM heatmap with shape (H, W) and
              values in the range [0, 1].
    """
    if not isinstance(sample, np.ndarray):
        raise TypeError("Sample must be a NumPy array.")

    if sample.ndim == 3:
        img_array = np.expand_dims(sample, axis=0)
        original_img = sample
    elif sample.ndim == 4 and sample.shape[0] == 1:
        img_array = sample
        original_img = sample[0]
    else:
        raise ValueError(
            "Sample must have shape (H, W, C) or (1, H, W, C)."
        )

    if preprocess_fn is not None:
        img_array = preprocess_fn(img_array.copy())

    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv_layer_name(model)

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model([img_array], training=False)

        if label_to_explain is None:
            if preds.shape[-1] == 1:
                label_to_explain = 0
                explanation_label_index = 1
            else:
                label_to_explain = int(tf.argmax(preds[0]))
                explanation_label_index = label_to_explain

        class_channel = preds[:, label_to_explain]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    if grads is None:
        raise ValueError(
            "Could not compute gradients. Make sure the selected layer is "
            "connected to the model output."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = last_conv_layer_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)

    if max_value == 0:
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap = heatmap / max_value

    heatmap = heatmap.numpy()

    heatmap_uint8 = np.uint8(255 * heatmap)
    colormap = mpl.colormaps[cmap]
    colormap_colors = colormap(np.arange(256))[:, :3]
    colorized_heatmap = colormap_colors[heatmap_uint8]

    colorized_heatmap = tf.keras.utils.array_to_img(colorized_heatmap)
    colorized_heatmap = colorized_heatmap.resize(
        (original_img.shape[1], original_img.shape[0])
    )
    colorized_heatmap = tf.keras.utils.img_to_array(colorized_heatmap)

    display_img = original_img.astype("float32")

    if display_img.shape[-1] == 1:
        display_img = np.repeat(display_img, 3, axis=-1)

    if display_img.max() <= 1.0:
        display_img = display_img * 255.0

    # Resize the normalized heatmap to image size so it can control opacity.
    # Low-importance pixels get alpha 0, so they do not show up as blue.
    heatmap_resized = tf.keras.utils.array_to_img(heatmap[..., np.newaxis])
    heatmap_resized = heatmap_resized.resize(
        (original_img.shape[1], original_img.shape[0])
    )
    heatmap_resized = tf.keras.utils.img_to_array(heatmap_resized).squeeze()
    heatmap_resized = heatmap_resized / 255.0

    alpha_mask = np.where(
        heatmap_resized >= importance_threshold,
        alpha,
        0.0
    )
    alpha_mask = alpha_mask[..., np.newaxis]

    superimposed_img = (
        colorized_heatmap * alpha_mask
        + display_img * (1 - alpha_mask)
    )
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")

    explanation_label = label_to_explain
    if class_names is not None:
        if preds.shape[-1] == 1 and len(class_names) == 2:
            explanation_label = class_names[explanation_label_index]
        else:
            explanation_label = class_names[label_to_explain]

    title_string = f'Grad-CAM explanation for "{explanation_label}"'

    if actual_label is not None:
        actual_label_display = actual_label
        if class_names is not None:
            actual_label_display = class_names[actual_label]
        title_string += f" (Actual label: {actual_label_display})"

    if save_figure or show:
        plt.imshow(superimposed_img)
        plt.title(title_string)
        plt.axis("off")

        if save_figure:
            plt.savefig(figure_save_path, bbox_inches="tight", pad_inches=0.1)

        if show:
            plt.show()
        else:
            plt.close()

    if return_heatmap:
        return superimposed_img, heatmap

    return superimposed_img
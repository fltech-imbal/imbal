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
    output_index=0,
    actual_value=None,
    preprocess_fn=None,
    alpha=0.4,
    importance_threshold=0.5,
    positive_cmap="Reds",
    negative_cmap="Blues",
    figure_save_path="gradcam-regression-explanation.png",
    save_figure=False,
    show=True,
    return_heatmap=False,
):
    """
    Generates a Grad-CAM explanation for a single image regression sample.

    This follows the same overall structure as the Keras Grad-CAM tutorial,
    but instead of explaining a class score, it explains a scalar regression
    output from the model.

    Args:
        sample: A single image as a NumPy array, shaped (H, W, C) or
            (1, H, W, C).
        model: A Keras-compatible image regression model.
        last_conv_layer_name: Optional name of the convolutional layer to
            explain. If None, the last Conv2D layer is used.
        output_index: Regression output index to explain. For standard
            single-output regression models, use 0.
        actual_value: Optional true regression value, displayed in the plot
            title.
        preprocess_fn: Optional preprocessing function applied before
            prediction.
        alpha: Maximum heatmap overlay opacity.
        importance_threshold: Minimum normalized importance required for a
            region to appear in the overlay.
        positive_cmap: Matplotlib colormap name used for regions that
            increase the predicted value.
        negative_cmap: Matplotlib colormap name used for regions that
            decrease the predicted value.
        figure_save_path: Path where the figure is saved if save_figure=True.
        save_figure: Whether to save the visualization.
        show: Whether to display the visualization.
        return_heatmap: Whether to return the raw heatmaps too.

    Returns:
        If return_heatmap is False:
            tuple[numpy.ndarray, numpy.ndarray]:
                A tuple containing:

                - positive_superimposed_img:
                  RGB Grad-CAM visualization image highlighting regions
                  associated with large increases in the predicted value,
                  with shape (H, W, 3) and dtype uint8.

                - negative_superimposed_img:
                  RGB Grad-CAM visualization image highlighting regions
                  associated with large decreases in the predicted value,
                  with shape (H, W, 3) and dtype uint8.

        If return_heatmap is True:
            tuple[
                numpy.ndarray,
                numpy.ndarray,
                numpy.ndarray,
                numpy.ndarray,
            ]:
                A tuple containing:

                - positive_superimposed_img:
                  RGB Grad-CAM visualization image highlighting regions
                  associated with large increases in the predicted value,
                  with shape (H, W, 3) and dtype uint8.

                - negative_superimposed_img:
                  RGB Grad-CAM visualization image highlighting regions
                  associated with large decreases in the predicted value,
                  with shape (H, W, 3) and dtype uint8.

                - positive_heatmap:
                  Normalized Grad-CAM heatmap representing regions that
                  positively contribute to the regression output, with
                  shape (H, W) and values in the range [0, 1].

                - negative_heatmap:
                  Normalized Grad-CAM heatmap representing regions that
                  negatively contribute to the regression output, with
                  shape (H, W) and values in the range [0, 1].
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

        # For regression, the model output itself is the target y-value.
        # For classification, this would normally be the pre-softmax score.
        if preds.shape.rank == 1:
            regression_output = preds
            predicted_value = float(preds[0])
        else:
            if output_index < 0 or output_index >= preds.shape[-1]:
                raise ValueError(
                    f"output_index={output_index} is out of bounds for model "
                    f"output shape {preds.shape}."
                )
            regression_output = preds[:, output_index]
            predicted_value = float(preds[0, output_index])

    grads = tape.gradient(regression_output, last_conv_layer_output)

    if grads is None:
        raise ValueError(
            "Could not compute gradients. Make sure the selected layer is "
            "connected to the model output."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = last_conv_layer_output[0]

    # Original Grad-CAM computes:
    #     sum_k alpha_k * A_k
    #
    # The previous version immediately applied ReLU, which only kept positive
    # influence. Here we keep the signed map first so that we can separate:
    #     positive values -> large increases
    #     negative values -> large decreases
    raw_heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    raw_heatmap = tf.squeeze(raw_heatmap).numpy()

    positive_heatmap = np.maximum(raw_heatmap, 0)
    negative_heatmap = np.maximum(-raw_heatmap, 0)

    positive_max = np.max(positive_heatmap)
    negative_max = np.max(negative_heatmap)

    if positive_max == 0:
        positive_heatmap = np.zeros_like(positive_heatmap)
    else:
        positive_heatmap = positive_heatmap / positive_max

    if negative_max == 0:
        negative_heatmap = np.zeros_like(negative_heatmap)
    else:
        negative_heatmap = negative_heatmap / negative_max

    display_img = original_img.astype("float32")

    if display_img.shape[-1] == 1:
        display_img = np.repeat(display_img, 3, axis=-1)

    if display_img.max() <= 1.0:
        display_img = display_img * 255.0

    original_display_img = np.clip(display_img, 0, 255).astype("uint8")

    def make_superimposed_img(heatmap, cmap):
        heatmap_uint8 = np.uint8(255 * heatmap)
        colormap = mpl.colormaps[cmap]
        colormap_colors = colormap(np.arange(256))[:, :3]
        colorized_heatmap = colormap_colors[heatmap_uint8]

        colorized_heatmap = tf.keras.utils.array_to_img(colorized_heatmap)
        colorized_heatmap = colorized_heatmap.resize(
            (original_img.shape[1], original_img.shape[0])
        )
        colorized_heatmap = tf.keras.utils.img_to_array(colorized_heatmap)

        heatmap_resized = tf.keras.utils.array_to_img(heatmap[..., np.newaxis])
        heatmap_resized = heatmap_resized.resize(
            (original_img.shape[1], original_img.shape[0])
        )
        heatmap_resized = tf.keras.utils.img_to_array(heatmap_resized).squeeze()
        heatmap_resized = heatmap_resized / 255.0

        alpha_mask = np.where(
            heatmap_resized >= importance_threshold,
            alpha,
            0.0,
        )
        alpha_mask = alpha_mask[..., np.newaxis]

        superimposed_img = (
            colorized_heatmap * alpha_mask
            + display_img * (1 - alpha_mask)
        )
        superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")

        return superimposed_img

    positive_superimposed_img = make_superimposed_img(
        positive_heatmap,
        positive_cmap,
    )

    negative_superimposed_img = make_superimposed_img(
        negative_heatmap,
        negative_cmap,
    )

    title_string = (
        "Grad-CAM regression explanation "
        f"for predicted value: {predicted_value:.4f}"
    )

    if actual_value is not None:
        title_string += f" (Actual value: {actual_value})"

    if save_figure or show:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].imshow(original_display_img)
        axes[0].set_title("Original image")
        axes[0].axis("off")

        axes[1].imshow(positive_superimposed_img)
        axes[1].set_title("Large increases in value")
        axes[1].axis("off")

        axes[2].imshow(negative_superimposed_img)
        axes[2].set_title("Large decreases in value")
        axes[2].axis("off")

        positive_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        positive_sm = mpl.cm.ScalarMappable(
            norm=positive_norm,
            cmap=positive_cmap,
        )
        positive_sm.set_array([])

        positive_cbar = fig.colorbar(
            positive_sm,
            ax=axes[1],
            fraction=0.046,
            pad=0.04,
        )
        positive_cbar.set_label("Increase intensity")

        negative_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        negative_sm = mpl.cm.ScalarMappable(
            norm=negative_norm,
            cmap=negative_cmap,
        )
        negative_sm.set_array([])

        negative_cbar = fig.colorbar(
            negative_sm,
            ax=axes[2],
            fraction=0.046,
            pad=0.04,
        )
        negative_cbar.set_label("Decrease intensity")

        fig.suptitle(title_string)

        fig.subplots_adjust(top=0.82, wspace=0.25)

        if save_figure:
            plt.savefig(figure_save_path, bbox_inches="tight", pad_inches=0.1)

        if show:
            plt.show()
        else:
            plt.close(fig)

    if return_heatmap:
        return (
            positive_superimposed_img,
            negative_superimposed_img,
            positive_heatmap,
            negative_heatmap,
        )

    return positive_superimposed_img, negative_superimposed_img
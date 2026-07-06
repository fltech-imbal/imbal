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


def _layer_uses_softmax(layer):
    if isinstance(layer, tf.keras.layers.Softmax):
        return True

    activation = getattr(layer, "activation", None)
    return activation == tf.keras.activations.softmax


def _find_pre_softmax_target(model, label_to_explain):
    """
    Returns a tensor for the class score before softmax when possible.

    Grad-CAM for softmax classification should use the score/logit before
    softmax as the numerator, not the probability after softmax.
    """
    output_layer = model.layers[-1]

    if isinstance(output_layer, tf.keras.layers.Softmax):
        pre_softmax_output = output_layer.input
        return pre_softmax_output[:, label_to_explain]

    if _layer_uses_softmax(output_layer):
        if not isinstance(output_layer, tf.keras.layers.Dense):
            raise ValueError(
                "The final layer uses softmax, but it is not a Dense layer "
                "or a separate Softmax layer. Please build the model with "
                "a separate pre-softmax/logits layer followed by Softmax."
            )

        previous_output = output_layer.input
        weights = output_layer.kernel
        bias = output_layer.bias

        pre_softmax_output = tf.keras.ops.matmul(previous_output, weights)

        if bias is not None:
            pre_softmax_output = tf.keras.ops.add(pre_softmax_output, bias)

        return pre_softmax_output[:, label_to_explain]
    else:
        raise ValueError(
            "We are expecting an output layer with softmax when there are 2 or more output units."
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
            predicted class is used. For binary one-output models, class 1 is
            explained by increasing the output and class 0 is explained by
            decreasing the output.
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

        If :code:`return_heatmap` is False, returns a NumPy array containing
        a superimposed RGB Grad-CAM visualization image with shape (H, W, 3) and dtype uint8.
        If :code:`return_heatmap` is True, returns a tuple :code:`(image, map)`, where :code:`image`
        is the superimposed RGB image, and :code:`map` is a NumPy array containing a normalized
        Grad-CAM heatmap with shape (H, W) and values in the range :math:`[0, 1]`.

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

    prediction_model = tf.keras.models.Model(model.inputs, model.output)
    preds = prediction_model([img_array], training=False)

    if label_to_explain is None:
        if preds.shape[-1] == 1:
            predicted_output = float(preds[0, 0])
            label_to_explain = int(predicted_output >= 0.5)
        else:
            label_to_explain = int(tf.argmax(preds[0]))

    if preds.shape[-1] == 1 and label_to_explain not in (0, 1):
        raise ValueError(
            "For one-output binary classification models, label_to_explain "
            "must be 0 or 1."
        )

    if preds.shape[-1] != 1 and (
        label_to_explain < 0 or label_to_explain >= preds.shape[-1]
    ):
        raise ValueError(
            f"label_to_explain={label_to_explain} is out of bounds for "
            f"model output shape {preds.shape}."
        )

    if preds.shape[-1] == 1:
        target_output = model.output[:, 0]

        if label_to_explain == 0:
            # For one-output binary classification, class 0 means evidence
            # that decreases the output score/probability for class 1.
            target_output = -target_output
    else:
        target_output = _find_pre_softmax_target(model, label_to_explain)

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, target_output],
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, pre_softmax_class_score = grad_model(
            [img_array],
            training=False,
        )

    score_gradients = tape.gradient(pre_softmax_class_score, last_conv_layer_output)

    if score_gradients is None:
        raise ValueError(
            "Could not compute gradients. Make sure the selected layer is "
            "connected to the model output."
        )

    pooled_grads = tf.reduce_mean(score_gradients, axis=(0, 1, 2))

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

    # Keep a clean copy of the original image for side-by-side display.
    original_display_img = np.clip(display_img, 0, 255).astype("uint8")

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
        explanation_label = class_names[label_to_explain]

    title_string = f'Grad-CAM explanation for "{explanation_label}"'

    if preds.shape[-1] == 1:
        if label_to_explain == 1:
            title_string += " — features supporting class 1"
        else:
            title_string += " — features supporting class 0"

    if actual_label is not None:
        actual_label_display = actual_label
        if class_names is not None:
            actual_label_display = class_names[actual_label]
        title_string += f" (Actual label: {actual_label_display})"

    if save_figure or show:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(original_display_img)
        axes[0].set_title("Original image")
        axes[0].axis("off")

        axes[1].imshow(superimposed_img)
        axes[1].set_title("Grad-CAM highlights")
        axes[1].axis("off")

        # Add a colorbar showing Grad-CAM importance intensity.
        # Using a dedicated axis keeps the colorbar outside the image.
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        # Reserve space on the right side for the colorbar.
        fig.subplots_adjust(right=0.88)

        # Create a dedicated axis for the colorbar.
        cbar_ax = fig.add_axes([0.90, 0.20, 0.02, 0.60])

        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Importance intensity")

        fig.suptitle(title_string)
        fig.subplots_adjust(top=0.88, right=0.88, wspace=0.15)

        if save_figure:
            plt.savefig(figure_save_path, bbox_inches="tight", pad_inches=0.1)

        if show:
            plt.show()
        else:
            plt.close(fig)

    if return_heatmap:
        return superimposed_img, heatmap

    return superimposed_img

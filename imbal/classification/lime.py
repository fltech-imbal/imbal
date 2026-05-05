from lime import lime_image
from matplotlib import pyplot as plt
import numpy as np
import imbal.util.backend.explanation as explanation

def lime_explain_image_sample(
        image,
        model,
        num_samples=100,
        num_features=5,
        class_names=None,
        actual_label=None,
        label_to_explain=None,
        save_figure=None
):
    """
    Utilizes LIME to generate an explanation for the classification of a particular image
    by a given model. For more about LIME, see :doc:`this page </imbal/lime-explanation>`.

    Args:
        image: The image to generate a LIME explanation for.
        model: The TensorFlow model to generate a LIME explanation from.
        num_samples: Optional, default 100. The number of local samples to perform for
            the LIME local approximation. See `LIME documentation <https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image>`_.
        num_features: Optional, default 5. The maximum number of features to present
            in the explanation. See `LIME documentation <https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image>`_.
        class_names: Optional, default :code:`None`. An array of strings, which maps
            class labels (as integer indices) to class names. If specified, is used to label the
            generated figure.
        actual_label: Optional, default :code:`None`. The true label of the
            provided image. If specified, is used only to label the generated figure.
        label_to_explain: Optional, default :code:`None`. The label of the class
            you wish to generate an explanation for. When set to :code:`None`, the
            label that is predicted by the model will be explained. This label need not be the same
            as the true label for the provided image.
        save_figure: Optional, default :code:`None`. If set to a string, the
            resultant plot with be saved to the specified path.

    Returns:
        :code:`None`
    """
    if len(image.shape) < 2 or len(image.shape) > 3:
        raise ValueError('"image" must be a 2D or 3D array (height, width, channels)')

    original_image = image
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    if len(image.shape) == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, -1)

    def predict_fn(value):
        if original_image.shape != value.shape[1:]:
            if original_image.ndim == 2 or original_image.ndim == 3 and original_image.shape[-1] == 1:
                value = value[..., 0]
            value = value.reshape((value.shape[0], *original_image.shape))
        return model.predict(value)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image,
        predict_fn,
        labels=[label_to_explain] if label_to_explain is not None else None,
        top_labels=1,
        num_samples=num_samples,
    )

    explanation_label_display = label_to_explain
    if explanation_label_display is None:
        explanation_label_display = explanation.top_labels[0]
    if class_names is not None:
        explanation_label_display = class_names[explanation_label_display]
    explanation_display = 'Prediction' if label_to_explain is None else 'Explanation'
    explanation_display += f' ({explanation_label_display})'

    if label_to_explain is None:
        label_to_explain = explanation.top_labels[0]

    result_image, mask = explanation.get_image_and_mask(
        label_to_explain,
        num_features=num_features
    )

    actual_label_display = ""
    if actual_label is not None:
        if class_names is not None:
            actual_label_display = class_names[actual_label]
        else:
            actual_label_display = actual_label


    result_image[:, :, 1][mask==1] = ((result_image[:, :, 1] + mask) / 2)[mask==1]

    fig, ax = plt.subplots(nrows=1, ncols=2)

    if class_names is not None:
        fig.suptitle(f'Explanation for "{explanation_label_display}"')
    ax[0].imshow(image)
    ax[0].set_title(f'Original Image{f" ({actual_label_display})" if actual_label is not None else ""}')
    ax[0].set_axis_off()
    ax[1].imshow(result_image)
    ax[1].set_title(explanation_display)
    ax[1].set_axis_off()

    if save_figure is not None:
        plt.savefig(save_figure)

    plt.show()

def lime_explain_tabular_sample(
        sample,
        model,
        training_data,
        num_samples=100,
        class_names=None,
        feature_names=None,
        actual_label=None,
        label_to_explain=None,
        figure_save_path='lime-explanation.html'
):
    """
    Utilizes LIME to generate an explanation for the classification of a particular sample
    by a given model. For more about LIME, see :doc:`this page </imbal/lime-explanation>`.

    Args:
        sample: The sample to generate a LIME explanation for.
        model: The TensorFlow model to generate a LIME explanation from.
        training_data: The data the given model was trained on.
        num_samples: Optional, default 100. The number of local samples to perform for
            the LIME local approximation. See `LIME documentation <https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image>`_.
        class_names: Optional, default :code:`None`. An array of strings, which maps
            class labels (as integer indices) to class names. If specified, is used to label the
            generated figure.
        feature_names: Optional, default :code:`None`. An array of strings, which maps
            features (by integer index) to feature names. If specified, is used to label the
            generated figure.
        actual_label: Optional, default :code:`None`. The true label of the
            provided image. If specified, is used only to label the generated figure.
        label_to_explain: Optional, default :code:`None`. The label of the class
            you wish to generate an explanation for. When set to :code:`None`, the
            label that is predicted by the model will be explained. This label need not be the same
            as the true label for the provided image.
        figure_save_path: Optional, default :code:`"lime-explanation.html"`. The path to
            save the generated HTML figure to.

    Returns:
        The path that the resulting figure was saved to, as a string.
    """
    return explanation.lime_explain_tabular_sample(
        sample,
        model,
        training_data,
        num_samples=num_samples,
        class_names=class_names,
        feature_names=feature_names,
        label_to_explain=label_to_explain,
        actual_label=actual_label,
        figure_save_path=figure_save_path,
        mode='classification',
    )
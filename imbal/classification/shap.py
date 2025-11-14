from lime import lime_image, lime_tabular
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np
import imbal.util.explanation as explanation


def shap_explain_image_sample(
        image,
        model,
        num_samples=100,
        num_features=100000,
        class_names=None,
        actual_label=None,
        label_to_explain=None,
        return_figure=False
):
    """
    Utilizes SHAP to generate an explanation for the classification of a particular image
    by a given model. For more about SHAP, see :doc:`this page </imbal/shap-explanation>`.

    Args:
        image: The image to generate a LIME explanation for.
        model: The PyTorch model to generate a LIME explanation from.
        num_samples: Optional, default 100. The number of local samples to perform for
            the LIME local approximation. See `LIME documentation <https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image>`_.
        num_features: Optional, default 100000. The maximum number of features to present
            in the explanation. See `LIME documentation <https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image>`_.
        class_names: Optional, default :code:`None`. An array of strings, which maps
            class labels (as integer indices) to class names. Used to label the
            generated figure.
        actual_label: Optional, default :code:`None`. default The true label of the
            provided image. Used to label the generated figure.
        label_to_explain: Optional, default :code:`None`. The label of the class
            you wish to generate an explanation for. This label need not be the same
            as the true label for the provided image. When set to :code:`None`, the
            label that is predicted by the model will be explained.
        return_figure: Optional, default :code:`False`. When set to :code:`True`, the
            Matplotlib Figure and Axes objects associated with the generated figure will
            be returned.

    Returns:
        :code:`None`, or a tuple :code:`(fig, ax)` containing a MatPlotLib Figure and Axes object, if
        :code:`return_figure` is set to :code:`True`.
    """
    if len(image.shape) < 2 or len(image.shape) > 3:
        raise ValueError('"image" must be a 2D or 3D array (height, width, channels)')

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    if len(image.shape) == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, -1)

    def predict_fn(value):
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

    temp, mask = explanation.get_image_and_mask(
        label_to_explain,
        positive_only=False,
        num_features=num_features,
        hide_rest=False
    )

    actual_label_display = ""
    if actual_label is not None:
        if class_names is not None:
            actual_label_display = class_names[actual_label]
        else:
            actual_label_display = actual_label
    fig, ax = plt.subplots(nrows=1, ncols=2)

    if class_names is not None:
        fig.suptitle(f'Explanation for "{explanation_label_display}"')
    ax[0].imshow(image)
    ax[0].set_title(f'Original Image{f" ({actual_label_display})" if actual_label is not None else ""}')
    ax[0].set_axis_off()
    ax[1].imshow(mark_boundaries(temp, mask))
    ax[1].set_title(explanation_display)
    ax[1].set_axis_off()

    if return_figure:
        return fig, ax

    plt.show()
    return None

def shap_explain_tabular_sample(
    sample,
    model,
    training_data,
    class_names=None,
    feature_names=None,
    label_to_explain=None,
    actual_label=None,
    plot_type='bar',
    figure_save_path='shap-explanation.png',
    save_figure=False,
    show=True
):
    """
    Utilizes SHAP to generate an explanation for the classification of a particular sample
    by a given model. For more about SHAP, see :doc:`this page </imbal/shap-explanation>`.

    Args:
        sample: The sample to generate a SHAP explanation for.
        model: The PyTorch model to generate a SHAP explanation from.
        training_data: The data the given model was trained on.
        class_names: Optional, default :code:`None`. An array of strings, which maps
            class labels (as integer indices) to class names. Used to label the
            generated figure.
        feature_names: Optional, default :code:`None`. An array of strings, which maps
            features (by integer index) to feature names. Used to label the
            generated figure.
        actual_label: Optional, default :code:`None`. The actual label for the sample
            being explained. Used to label the generated figure.
        label_to_explain: Optional, default :code:`None`. The label of the class
            you wish to generate an explanation for. This label need not be the same
            as the true label for the provided image. When set to :code:`None`, the
            label that is predicted by the model will be explained.
        save_figure: Optional, default :code:`False`. Whether to save the generated figure.
        figure_save_path: Optional, default :code:`"shap-explanation.png"`. The path to
            save the generated figure to.
        plot_type: Optional, default :code:`"bar"`. The type of plot to generate. Available options are
            :code:`"bar"` and :code:`"waterfall"`. See `SHAP documentation <https://shap.readthedocs.io/en/latest/api_examples.html#plots>`_.
        show: Optional, default :code:`True`. Whether to show the generated figure. If set to
            :code:`False`, the figure can be further modified before displaying or saving it.

    Returns:
        None
    """
    return explanation.shap_explain_tabular_sample(
        sample,
        model,
        training_data,
        label_to_explain=label_to_explain,
        class_names=class_names,
        feature_names=feature_names,
        figure_save_path=figure_save_path,
        actual_label=actual_label,
        show=show,
        plot_type=plot_type,
        save_figure=save_figure,
        mode='classification'
    )

def shap_explain_tabular_dataset(
    dataset,
    model,
    training_data,
    label_to_explain,
    class_names=None,
    feature_names=None,
    plot_type='bar',
    figure_save_path='shap-explanation.png',
    save_figure=False,
    show = True,
):
    """
    Utilizes SHAP to generate an explanation for the classification of a particular dataset
    by a given model. For more about SHAP, see :doc:`this page </imbal/shap-explanation>`.

    Args:
        dataset: The dataset to generate a SHAP explanation for.
        model: The PyTorch model to generate a SHAP explanation from.
        training_data: The data the given model was trained on.
        class_names: Optional, default :code:`None`. An array of strings, which maps
            class labels (as integer indices) to class names. Used to label the
            generated figure.
        label_to_explain: The label of the class
            you wish to generate an explanation for. This label need not be the same
            as the true label for the provided image.
        feature_names: Optional, default :code:`None`. An array of strings, which maps
            features (by integer index) to feature names. Used to label the
            generated figure.
        save_figure: Optional, default :code:`False`. Whether to save the generated figure.
        figure_save_path: Optional, default :code:`"shap-explanation.png"`. The path to
            save the generated figure to.
        plot_type: Optional, default :code:`"heatmap"`. The type of plot to generate. Available options are
            :code:`"heatmap"`, :code:`"beeswarm"`, and :code:`"violin"`. See `SHAP documentation <https://shap.readthedocs.io/en/latest/api_examples.html#plots>`_.
        show: Optional, default :code:`True`. Whether to show the generated figure. If set to
            :code:`False`, the figure can be further modified before displaying or saving it.

    Returns:
        None
    """
    return explanation.shap_explain_tabular_dataset(
        dataset,
        model,
        training_data,
        label_to_explain,
        class_names=class_names,
        feature_names=feature_names,
        figure_save_path=figure_save_path,
        show=show,
        plot_type=plot_type,
        save_figure=save_figure,
        mode='classification'
    )
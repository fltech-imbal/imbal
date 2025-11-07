from lime import lime_image, lime_tabular
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np
import imbal.util.explanation as explanation

def lime_image_explanation(
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
        Utilizes LIME to generate an explanation for the classification of a particular image
        by a given model. For more about LIME, see :doc:`this page </imbal/lime-explanation>`.
    
        Args:
            image: The image to generate a LIME explanation for.
            model: The PyTorch model whose representation space you wish to visualize.
            num_samples: TODO
            num_features: TODO
            class_names: TODO
            actual_label: TODO
            label_to_explain: TODO
            return_figure: TODO

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
        top_labels=1 if label_to_explain is None else None,
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

def lime_tabular_explanation(
        image,
        model,
        training_data,
        num_samples=100,
        class_names=None,
        feature_names=None,
        label=None,
        figure_save_path='temp.html',
        use_pyplot=False,
        return_figure=False,
):
    """

    """
    return explanation.lime_tabular_explanation(
        image,
        model,
        training_data,
        num_samples=num_samples,
        class_names=class_names,
        feature_names=feature_names,
        label=label,
        figure_save_path=figure_save_path,
        use_pyplot=use_pyplot,
        return_figure=return_figure,
        mode='classification',
    )
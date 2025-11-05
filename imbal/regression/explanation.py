from lime import lime_image, lime_tabular
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

def lime_image_explanation(
        image,
        model,
        num_samples=100,
        num_features=1000,
        label=None,
        return_figure=False,
        explained_feature_index=None
):
    if len(image.shape) < 2 or len(image.shape) > 3:
        raise ValueError('"image" must be a 2D or 3D array (height, width, channels)')

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    if len(image.shape) == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, -1)

    def predict_fn(value):
        result = model.predict(value)
        if explained_feature_index is None:
            print('result', result)
            return result
        else:
            return result[explained_feature_index]

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image,
        predict_fn,
        labels=[label],
        top_labels=1 if label is None else None,
        hide_color=None,
        num_samples=num_samples,
    )

    if label is None:
        label = explanation.top_labels[0]

    temp, mask = explanation.get_image_and_mask(
        label,
        positive_only=False,
        num_features=num_features,
        hide_rest=False
    )

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].set_axis_off()
    ax[1].imshow(mark_boundaries(temp, mask))
    ax[1].set_title('Explanation')
    ax[1].set_axis_off()

    if return_figure:
        return fig, ax

    plt.show()
    return None

def lime_tabular_explaination(
        image,
        model,
        training_data,
        num_samples=100,
        feature_names=None,
        label=None,
        figure_save_path='temp.html',
        use_pyplot=False,
        return_figure=False,
):

    def predict_fn(value):
        return model.predict(value)


    explainer = lime_tabular.LimeTabularExplainer(
        training_data,
        mode='regression',
        feature_names=feature_names,
    )
    explanation = explainer.explain_instance(
        image,
        predict_fn,
        labels=[label],
        top_labels=1 if label is None else None,
        num_samples=num_samples,
    )

    if use_pyplot:
        fig = explanation.as_pyplot_figure()
        plt.show()
        if return_figure:
            return fig
    else:
        explanation.save_to_file(figure_save_path)

    return None


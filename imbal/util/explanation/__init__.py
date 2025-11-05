from lime import lime_image, lime_tabular
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
import tensorflow as tf
import numpy as np

def lime_explanation(
    data,
    labels,
    model,
    instance_index=0,
    segmentation_fn=None,
    lime_mode='image',
    model_type='classification',
    num_samples=100,
    num_features=4,
    top_labels=3
):
    if lime_mode == 'image':
        _lime_explain_image(
            data,
            model,
            instance_index=instance_index,
            segmentation_fn=segmentation_fn,
            model_type=model_type,
            num_samples=num_samples,
            num_features=num_features,
            top_labels=top_labels
        )
    elif lime_mode == 'tabular':
        _lime_explain_tabular(
            data,
            labels,
            model,
            instance_index=instance_index,
            segmentation_fn=segmentation_fn,
            model_type=model_type,
            num_samples=num_samples,
            num_features=num_features,
            top_labels=top_labels
        )
    else:
        raise ValueError('lime_mode must be either "image" or "tabular"')

def _lime_explain_image(
        data,
        model,
        instance_index=0,
        segmentation_fn=None,
        model_type='classification',
        num_samples=100,
        num_features=4,
        top_labels=3
):
    def predict_fn(images):
        gray = tf.image.rgb_to_grayscale(images)
        gray = tf.image.resize(gray, [28, 28])
        return model.predict(gray)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        data[instance_index],
        predict_fn,
        top_labels=top_labels,
        hide_color=None,
        num_samples=num_samples,
        segmentation_fn=segmentation_fn,
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=num_features,
        hide_rest=False
    )

    print(explanation.top_labels[0])
    plt.imshow(mark_boundaries(temp, mask))
    plt.axis('off')
    plt.show()

def _lime_explain_tabular(
        data,
        labels,
        model,
        instance_index=0,
        segmentation_fn=None,
        model_type='classification',
        num_samples=100,
        num_features=4,
        top_labels=3
):

    explainer = lime_tabular.LimeTabularExplainer(
        data,
        mode=model_type,
        training_labels=labels,
    )

    def predict_fn(value):
        p1 = model.predict(value).reshape(-1)  # shape (n_samples,)
        p0 = 1 - p1
        return np.vstack([p0, p1]).T

    exp = explainer.explain_instance(
        data[instance_index],
        predict_fn,
        num_features=num_features
    )

    # Get HTML representation
    html_content = exp.as_html()

    # Save to file
    with open("lime_explanation.html", "w") as f:
        f.write(html_content)

    exp.as_pyplot_figure()
    plt.show()

def shap_explanation():
    pass
import imbal.util.explanation as explanation

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
    show=True
):
    """
    Utilizes LIME to generate an explanation for the classification of a particular sample
    by a given model. For more about LIME, see :doc:`this page </imbal/lime-explanation>`.

    Args:
        sample: The sample to generate a LIME explanation for.
        model: The PyTorch model to generate a LIME explanation from.
        training_data: The data the given model was trained on.
        num_samples: Optional, default 100. The number of local samples to perform for
            the LIME local approximation. See `LIME documentation <https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image>`_.
        class_names: Optional, default :code:`None`. An array of strings, which maps
            class labels (as integer indices) to class names. Used to label the
            generated figure.
        feature_names: Optional, default :code:`None`. An array of strings, which maps
            features (by integer index) to feature names. Use to label the
            generated figure.
        label_to_explain: Optional, default :code:`None`. The label of the class
            you wish to generate an explanation for. This label need not be the same
            as the true label for the provided image. When set to :code:`None`, the
            label that is predicted by the model will be explained.
        figure_save_path: Optional, default :code:`"lime-explanation.html"`. The path to
            save the generated HTML figure to.
        use_pyplot: Optional, default :code:`False`. Whether to use pyplot
            to display the generated figure. The `LIME documentation <https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image>`_
            recommends not using the pyplot plot, but it can be useful for quick visualization.
        return_figure: Optional, default :code:`False`. When set to :code:`True`, the
            Matplotlib Figure and Axes objects associated with the generated figure will
            be returned. Ignored when :code:`use_pyplot` is set to :code:`False`.

    Returns:
        :code:`None`, or a tuple :code:`(fig, ax)` containing a MatPlotLib Figure and Axes object, if
        :code:`return_figure` is set to :code:`True`.
    """
    return explanation.shap_explain_tabular_sample(
        sample,
        model,
        training_data,
        class_names=class_names,
        feature_names=feature_names,
        label_to_explain=label_to_explain,
        actual_label=actual_label,
        figure_save_path=figure_save_path,
        show=show,
        plot_type=plot_type,
        mode='regression'
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
    show=True,
):
    """
    Utilizes LIME to generate an explanation for the classification of a particular sample
    by a given model. For more about LIME, see :doc:`this page </imbal/lime-explanation>`.

    Args:
        sample: The sample to generate a LIME explanation for.
        model: The PyTorch model to generate a LIME explanation from.
        training_data: The data the given model was trained on.
        num_samples: Optional, default 100. The number of local samples to perform for
            the LIME local approximation. See `LIME documentation <https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image>`_.
        class_names: Optional, default :code:`None`. An array of strings, which maps
            class labels (as integer indices) to class names. Used to label the
            generated figure.
        feature_names: Optional, default :code:`None`. An array of strings, which maps
            features (by integer index) to feature names. Use to label the
            generated figure.
        label_to_explain: Optional, default :code:`None`. The label of the class
            you wish to generate an explanation for. This label need not be the same
            as the true label for the provided image. When set to :code:`None`, the
            label that is predicted by the model will be explained.
        figure_save_path: Optional, default :code:`"lime-explanation.html"`. The path to
            save the generated HTML figure to.
        use_pyplot: Optional, default :code:`False`. Whether to use pyplot
            to display the generated figure. The `LIME documentation <https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image>`_
            recommends not using the pyplot plot, but it can be useful for quick visualization.
        return_figure: Optional, default :code:`False`. When set to :code:`True`, the
            Matplotlib Figure and Axes objects associated with the generated figure will
            be returned. Ignored when :code:`use_pyplot` is set to :code:`False`.

    Returns:
        :code:`None`, or a tuple :code:`(fig, ax)` containing a MatPlotLib Figure and Axes object, if
        :code:`return_figure` is set to :code:`True`.
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
        mode='regression'
    )


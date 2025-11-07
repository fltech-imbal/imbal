import imbal.util.explanation as explanation

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
        mode='regression'
    )


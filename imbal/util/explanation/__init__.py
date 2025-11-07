from lime import lime_image, lime_tabular
from matplotlib import pyplot as plt

def lime_tabular_explanation(
        image,
        model,
        training_data,
        num_samples=100,
        class_names=None,
        feature_names=None,
        label=None,
        mode='classification',
        figure_save_path='lime-explanation.html',
        use_pyplot=False,
        return_figure=False,
):

    def predict_fn(value):
        return model.predict(value)

    explainer = lime_tabular.LimeTabularExplainer(
        training_data,
        mode=mode,
        class_names=class_names,
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
        print(f'LIME explanation saved to "{figure_save_path}"')

    return figure_save_path
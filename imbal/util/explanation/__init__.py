from lime import lime_tabular
import shap
import numpy as np
from matplotlib import pyplot as plt

def lime_explain_tabular_sample(
        sample,
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
        sample,
        predict_fn,
        labels=[label],
        top_labels=1 if label is None else None,
        num_samples=num_samples,
    )

    if use_pyplot:
        fig = explanation.as_pyplot_figure()
        if return_figure:
            return fig
        else:
            plt.show()
    else:
        explanation.save_to_file(figure_save_path)
        print(f'LIME explanation saved to "{figure_save_path}"')

    return figure_save_path

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
    show=True,
    mode='classification'
):
    explainer = shap.Explainer(model, training_data)
    shap_values = explainer(np.expand_dims(sample, axis=0))

    if label_to_explain is None:
        label_to_explain = model.predict(np.expand_dims(sample, axis=0))[0].argmax()

    if mode == 'classification':
        single_class_expl = shap.Explanation(
            values=shap_values.values[0][:, label_to_explain],
            base_values=shap_values.base_values[0][label_to_explain],
            data=shap_values.data[0],
            feature_names=feature_names,
            output_names=class_names
        )
    else:
        single_class_expl = shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=shap_values.data[0],
            feature_names=feature_names,
            output_names=class_names
        )

    if plot_type == 'bar':
        ax = shap.plots.bar(single_class_expl, show=False)
    elif plot_type == 'waterfall':
        ax = shap.plots.waterfall(single_class_expl, show=False)
    else:
        raise ValueError('Invalid plot type')

    explanation_label = label_to_explain
    if class_names is not None:
        explanation_label = class_names[label_to_explain]
    title_string = f'Explanation for "{explanation_label}"'

    if actual_label is not None:
        if class_names is not None:
            actual_label = class_names[actual_label]
        title_string += f' (Actual label: {actual_label})'

    ax.set_title(title_string)

    if save_figure:
        plt.savefig(figure_save_path)

    if show:
        plt.show()

def shap_explain_tabular_dataset(
    dataset,
    model,
    training_data,
    label_to_explain,
    class_names=None,
    feature_names=None,
    plot_type='heatmap',
    figure_save_path='shap-explanation.png',
    save_figure=False,
    show=True,
    mode='classification'
):
    explainer = shap.Explainer(model, training_data)
    shap_values = explainer(dataset)

    if label_to_explain is None:
        label_to_explain = model.predict(np.expand_dims(dataset, axis=0))[0].argmax()

    if mode == 'classification':
        single_class_expl = shap.Explanation(
            values=shap_values.values[:, :, label_to_explain],
            base_values=shap_values.base_values[:, label_to_explain],
            data=shap_values.data,
            feature_names=feature_names,
            output_names=class_names
        )
    else:
        single_class_expl = shap.Explanation(
            values=shap_values.values,
            base_values=shap_values.base_values,
            data=shap_values.data,
            feature_names=feature_names,
            output_names=class_names
        )

    if plot_type == 'beeswarm':
        shap.plots.beeswarm(single_class_expl, show=False)
    elif plot_type == 'violin':
        shap.plots.violin(single_class_expl, show=False)
    elif plot_type == 'heatmap':
        shap.plots.heatmap(single_class_expl, show=False)
    else:
        raise ValueError('Invalid plot type')


    plt.title(f'Explanation of dataset')

    if save_figure:
        plt.savefig(figure_save_path)

    if show:
        plt.show()
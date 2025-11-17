from lime import lime_tabular
import shap
import numpy as np
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup

def lime_explain_tabular_sample(
        sample,
        model,
        training_data,
        num_samples=100,
        class_names=None,
        feature_names=None,
        label_to_explain=None,
        actual_label=None,
        mode='classification',
        figure_save_path='lime-explanation.html'
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
        labels=[label_to_explain],
        top_labels=1 if label_to_explain is None else None,
        num_samples=num_samples,
    )

    if label_to_explain is None:
        label_to_explain = model.predict(np.expand_dims(sample, axis=0))[0]
        if mode == 'classification':
            label_to_explain = label_to_explain.argmax()
        else:
            label_to_explain = f'{label_to_explain[0]:.3f}'

    explanation_label = label_to_explain
    if class_names is not None:
        explanation_label = class_names[label_to_explain]
    title_string = f'Explanation for "{explanation_label}"'

    if actual_label is not None:
        if class_names is not None:
            actual_label = class_names[actual_label]
        title_string += f' (Actual label: {actual_label})'

    soup = BeautifulSoup(explanation.as_html(), "lxml")
    body = soup.find('body')
    title = soup.new_tag('h1')
    title.string = title_string
    title['style'] = 'text-align: center'
    body.insert(0, title)

    with open(figure_save_path, "w") as f:
        f.write(str(soup))
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
    if not isinstance(sample, np.ndarray):
        raise TypeError('Sample must be a Numpy array.')
    if not isinstance(training_data, np.ndarray):
        raise TypeError('Training data must be a Numpy array.')

    explainer = shap.Explainer(model, training_data)
    shap_values = explainer(np.expand_dims(sample, axis=0))

    if label_to_explain is None:
        label_to_explain = model.predict(np.expand_dims(sample, axis=0))[0]
        if mode == 'classification':
            label_to_explain = label_to_explain.argmax()
        else:
            label_to_explain = f'{label_to_explain[0]:.3f}'

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
    label_to_explain=None,
    class_names=None,
    feature_names=None,
    plot_type='heatmap',
    figure_save_path='shap-explanation.png',
    save_figure=False,
    show=True,
    mode='classification'
):
    if not isinstance(dataset, np.ndarray):
        raise TypeError('Dataset must be a Numpy array.')
    if not isinstance(training_data, np.ndarray):
        raise TypeError('Training data must be a Numpy array.')

    explainer = shap.Explainer(model, training_data)
    shap_values = explainer(dataset)

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


    if class_names is None:
        if label_to_explain is None:
            explanation_label = f'predictions'
        else:
            explanation_label = f'Class {label_to_explain}'
    else:
        explanation_label = f'class "{class_names[label_to_explain]}"'


    plt.title(f'Explanation of {explanation_label} across dataset')

    if save_figure:
        plt.savefig(figure_save_path)
    if show:
        plt.show()
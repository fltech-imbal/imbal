# shap_explain_tabular_sample

```{eval-rst}
.. autoclass:: imbal.classification.shap_explain_tabular_sample
```

Example:

```python
>>> # Assume a TensorFlow model is already saved to 'model'
>>>
>>> from sklearn.datasets import load_wine
>>> 
>>> x, y = load_wine(return_X_y=True)
>>> labels = load_wine().feature_names
>>>
>>> imbal.classification.shap_explain_tabular_sample(
>>>     x[0],
>>>     model,
>>>     x_train,
>>>     plot_type='waterfall',
>>>     class_names=['Region 1', 'Region 2', 'Region 3'],
>>>     feature_names=labels
>>> )
```

## Plot Examples

Below is an example of the waterfall plot for a correctly predicted sample in the `scikit-learn`
[wine dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html).

```python
>>> imbal.classification.shap_explain_tabular_sample(
>>>     x,
>>>     model,
>>>     x_train,
>>>     class_names=['Region 1', 'Region 2', 'Region 3'],
>>>     feature_names=labels,
>>>     plot_type='waterfall'
>>> )
```

<img 
style="width: 400px"
src="../../_static/classification/shap_tabular_sample/shap-explanation-0-waterfall.png"/>

Below is an example of the bar plot for an incorrectly predicted sample.

```python
>>> imbal.classification.shap_explain_tabular_sample(
>>>     x,
>>>     model,
>>>     x_train,
>>>     class_names=['Region 1', 'Region 2', 'Region 3'],
>>>     feature_names=labels,
>>>     plot_type='bar'
>>> )
```

<img 
style="width: 400px"
src="../../_static/classification/shap_tabular_sample/shap-explanation-4-bar.png"/>

Below is an example of a bar plot explanation of the correct class
for the incorrectly predicted sample shown above.

```python
>>> imbal.classification.shap_explain_tabular_sample(
>>>     x,
>>>     model,
>>>     x_train,
>>>     label_to_explain=y,
>>>     class_names=['Region 1', 'Region 2', 'Region 3'],
>>>     feature_names=labels,
>>>     plot_type='bar'
>>> )
```

<img 
style="width: 400px"
src="../../_static/classification/shap_tabular_sample/shap-explanation-override-4-bar.png"/>
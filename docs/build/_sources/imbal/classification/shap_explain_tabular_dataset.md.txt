# shap_explain_tabular_dataset

```{eval-rst}
.. autoclass:: imbal.classification.shap_explain_tabular_dataset
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
>>> imbal.classification.shap_explain_tabular_dataset(
>>>     x[0],
>>>     model,
>>>     x_train,
>>>     y[0],
>>>     class_names=['Region 1', 'Region 2', 'Region 3'],
>>>     feature_names=labels,
>>>     plot_type='beeswarm'
>>> )
```

## Plot Examples

Below is an example of a heatmap plot explaining a model's predictions
for the third class of the `scikit-learn`
[wine dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html).

```python
>>> imbal.classification.shap_explain_tabular_dataset(
>>>     x,
>>>     model,
>>>     x_train,
>>>     2, # Index 2 corresponds to class 3
>>>     class_names=['Region 1', 'Region 2', 'Region 3'],
>>>     feature_names=labels,
>>>     plot_type='heatmap'
>>> )
```

<img 
style="width: 400px"
src="../../_static/classification/shap_tabular_dataset/shap-explanation-2-heatmap.png"/>

Below is an example of a beeswarm plot explaining predictions
for the third class of the same dataset.

```python
>>> imbal.classification.shap_explain_tabular_dataset(
>>>     x,
>>>     model,
>>>     x_train,
>>>     2, # Index 2 corresponds to class 3
>>>     class_names=['Region 1', 'Region 2', 'Region 3'],
>>>     feature_names=labels,
>>>     plot_type='beeswarm'
>>> )
```

<img 
style="width: 400px"
src="../../_static/classification/shap_tabular_dataset/shap-explanation-2-beeswarm.png"/>

Below is an example of a violin plot explaining predictions
for the third class of the same dataset.

```python
>>> imbal.classification.shap_explain_tabular_dataset(
>>>     x,
>>>     model,
>>>     x_train,
>>>     2, # Index 2 corresponds to class 3
>>>     class_names=['Region 1', 'Region 2', 'Region 3'],
>>>     feature_names=labels,
>>>     plot_type='violin'
>>> )
```

<img 
style="width: 400px"
src="../../_static/classification/shap_tabular_dataset/shap-explanation-2-violin.png"/>
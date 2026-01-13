# lime_explain_tabular_sample

```{eval-rst}
.. autoclass:: imbal.regression.lime_explain_tabular_sample
```

Example:

```python
>>> # Assume a TensorFlow model is already saved to 'model'
>>>
>>> from sklearn.datasets import fetch_california_housing
>>> 
>>> x, y = fetch_california_housing(return_X_y=True)
>>> labels = fetch_california_housing().feature_names
>>>
>>> imbal.regression.lime_explain_tabular_sample(
>>>     x[0],
>>>     model,
>>>     x_train,
>>>     label_to_explain=y[0]
>>>     feature_names=labels
>>> )
```

## Plot Examples

Below is an example of the resulting HTML plot for a correctly predicted sample (within a reasonable tolerance) 
in the `scikit-learn`
[California housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html).

```python
>>> imbal.regression.lime_explain_tabular_sample(
>>>     x,
>>>     model,
>>>     x_train,
>>>     actual_label=y,
>>>     feature_names=labels
>>> )
```

<img 
style="width: 600px"
src="../../_static/regression/lime_tabular_explanation/lime_tabular_explanation_example.png"/>

Below is an example of the resulting HTML plot for an incorrectly predicted sample.

```python
>>> imbal.regression.lime_explain_tabular_sample(
>>>     x,
>>>     model,
>>>     x_train,
>>>     actual_label=y,
>>>     feature_names=labels
>>> )
```

<img 
style="width: 600px"
src="../../_static/regression/lime_tabular_explanation/lime_tabular_explanation_example_incorrect.png"/>

Below is an example of the resulting HTML plot for the explanation of the correct class
for the incorrectly predicted sample shown above. Note that the prediction shown is the same, despite the
value being explained being overridden. Still, the explained values have been updated to reflect the
desired override value.

```python
>>> imbal.regression.lime_explain_tabular_sample(
>>>     x,
>>>     model,
>>>     x_train,
>>>     label_to_explain=y,
>>>     actual_label=y,
>>>     feature_names=labels
>>> )
```

<img 
style="width: 600px"
src="../../_static/regression/lime_tabular_explanation/lime_tabular_explanation_example_overridden.png"/>
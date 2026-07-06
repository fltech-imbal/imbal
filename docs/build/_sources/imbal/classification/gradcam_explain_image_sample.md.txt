# gradcam_explain_image_sample

```{eval-rst}
.. autoclass:: imbal.classification.gradcam_explain_image_sample
```

Example:

```python
>>> # Assume a TensorFlow model is already saved to 'model'
>>>
>>> class_labels = ['airplane', 'bird', 'car', 'cat', 'deer',
>>>                 'dog', 'horse', 'monkey', 'ship', 'truck']
>>>
>>> x = x_test[0]
>>> y = y_test[0]
>>>
>>> fig = imbal.classification.gradcam_explain_image_sample(
>>>     x,
>>>     model,
>>>     actual_label=y,
>>>     label_to_explain=y,
>>>     class_names=class_labels
>>> )
```

## Plot Examples

### Correct Classification

Below is an example of the resulting Matplotlib pyplot plot for a correctly predicted class.

```python
>>> imbal.classification.gradcam_explain_image_sample(
>>>     x,
>>>     model,
>>>     actual_label=y
>>> )
```

<img
src="../../_static/classification/gradcam_image_classification/gradcam_image_explanation_correct_pred.png"
width="450px"/>

The left panel shows the original image.

The right panel shows the Grad-CAM explanation. Regions highlighted in red contribute toward increasing the predicted class probability, while regions highlighted in blue contribute little or none toward increasing the predicted class probability. Color intensity corresponds to the magnitude of each region's contribution.

The figure title displays both the predicted class and the actual target class. The colorbar indicates the intensity of the positive contribution to the prediction.

### Incorrect Classification

An example of the resulting Matplotlib pyplot plot for an incorrectly predicted class.

```python
>>> imbal.classification.gradcam_explain_image_sample(
>>>     x,
>>>     model,
>>>     actual_label=y
>>> )
```

<img src="../../_static/classification/gradcam_image_classification/gradcam_image_explanation_incorrect_pred.png"
width="450px"/>

This example demonstrates how Grad-CAM can be used to investigate the regions of an image that most strongly influenced a prediction, even when the predicted class is incorrect.

As with the previous example, red regions indicate image areas that increase the predicted class probability, while blue regions indicate areas contributing little or none toward increasing the predicted class probability. The intensity of the color reflects the relative contribution magnitude.
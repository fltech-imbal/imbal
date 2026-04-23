# Model representation visualization using t-SNE

Sometimes, it is useful to be able to visualize the representation
space of a machine learning model to better understand how data
is being represented by the model. Techniques such as [t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
allow for the representation space of a model to be visualized in an easy-to-understand
2D plot. This tutorial will explain how to use t-SNE within `imbal`, extending
the code from the [Balanced Fit with Validation](balanced_fit_val.md) tutorial.

All the source code in the tutorial can be found at `imbal/tutorials/SDO/regression/visualization.py`.

## Visualizing with t-SNE

First, we will use the code provided in the [Balanced Fit with Validation](balanced_fit_val.md) tutorial
to train a model on the SDOBenchmark dataset. Then, we can use
[imbal.classification.tsne_visualization](../../../regression/tsne_visualization.md)
function built into `imbal` to easily generate a t-SNE plot for our model.

By default, `imbal` will use the second to last layer of a neural network as
the layer to visualize using t-SNE. This means that the predictions made by
the model are the result of a linear combination of relations shown in the
t-SNE plot. Therefore, ideally, there exists a smooth gradient of label values
across the t-SNE plot.

We add the following code at the end of the original tutorial code:

```python
imbal.regression.tsne_visualization(
    model,
    x_test,
    y_test,
    save_figure='tsne-regression-visualization.png'
)
```
which results in the following plot:

<div style="display: flex; gap: 8px; max-width: 100%;">
<img style="flex:1; max-width: 49%;" src="../../../../_static/tutorials/SDO/tsne-regression-visualization.png"/>
</div>

As you can see, there is not a smooth gradient from the minimum to the maximum values
in the resulting t-SNE plot. This could be due to a number of factors, but a likely cause is that
the subset of the SDOBenchmark dataset used in the tutorials is highly reduced, meaning the model is
being trained on significantly less data than is provided by SDOBenchmark.
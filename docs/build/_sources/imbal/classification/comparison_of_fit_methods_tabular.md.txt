# Comparison of Methods: Regular vs. Balanced vs. Decoupled Fit

Below is a comparison of [decoupled fit](decoupled_fit.md) and [balanced fit](balanced_fit.md)
to a standard, unbalanced fit, and a weight-balanced
fit of the dataset shown below, which contains 1,531 data points.

### Data Distribution

<img src="../../_static/classification/decoupled_fit/sep-ec-kde-curve.png" width="500"/>

### Regular Fit

```python
    model.compile(
        loss="mse",
        optimizer=keras.optimizers.Adam(learning_rate=2e-5),
        metrics=["mse"]
    )
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs
    )
```
    
<div style="display: flex; max-width: 100%; width:650px">
<img alt="test"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/fit-comparison-.png"/>
<img alt="test 2"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/tsne_visualization-.png"/>
</div>

### Balanced Fit

```python
    compile_parameters = imbal.classification.wrap_model_compile_parameters(
        loss="mse",
        optimizer=keras.optimizers.Adam(learning_rate=2e-5),
        metrics=["mse"]
    )

    kde_fit_parameters = imbal.regression.wrap_kde_fit_parameters(
        bin_count=BIN_COUNT
    )
    
    imbal.classification.balanced_fit(
        model,
        x_train,
        y_train,
        compile_parameters=parameters,
        epochs=epochs,
        batch_size=batch_size
    )
```

<div style="display: flex; max-width: 100%; width:650px">
<img alt="test"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/fit-comparison-balanced.png"/>
<img alt="test 2"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/tsne_visualization-balanced.png"/>
</div>

### Decoupled Fit

```python
    compile_parameters = imbal.classification.wrap_model_compile_parameters(
        loss="mse",
        optimizer=keras.optimizers.Adam(learning_rate=2e-5),
        metrics=["mse"]
    )

    kde_fit_parameters = imbal.regression.wrap_kde_fit_parameters(
        bin_count=BIN_COUNT
    )
    
    imbal.regression.decoupled_fit(
        model,
        x_train,
        y_train,
        compile_parameters=compile_parameters,
        kde_fit_parameters=kde_fit_parameters,
        epochs=epochs,
        batch_size=batch_size,
        representation_layer_index=-3
    )
```

<div style="display: flex; max-width: 100%; width:650px">
<img alt="test"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/fit-comparison-decoupled.png"/>
<img alt="test 2"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/tsne_visualization-decoupled.png"/>
</div>

### Comparison of Performance

For the table below, common samples refer to samples whose label
falls in the range $[-1, 1]$, and rare samples are those whose label
falls outside of this range.

| Method    | Time (s) | Frequent Sample MSE | Rare Sample MSE |
|-----------|----------|---------------------|-----------------| 
| Regular   | $???$    | $7.5761$            | $0.6142$        |
| Balanced  | $???$    | $6.2100$            | $1.0090$        |
| Decoupled | $???$    | $1.4113$            | $0.9988$        |

See also: [Comparison of Autoencoder Methods](comparison_of_ae_methods_tabular.md)
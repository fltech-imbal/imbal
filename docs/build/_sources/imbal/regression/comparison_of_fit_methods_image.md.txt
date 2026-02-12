# Comparison of Methods: Regular vs. Balanced vs. rRT Fit on Image Regression

Below is a comparison of the [regression.Model object's](model.md) `balanced_fit`
and `decoupled_fit` functions, and a standard fit where instances are equally weighted,
on the [AgeDB dataset](https://ibug.doc.ic.ac.uk/resources/agedb/).

### Regular Fit

```python
    # Assume data has already been loaded into x_train, y_train, x_test, and y_test

    inputs = keras.Input(shape=(112, 88, 3))
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2, 2))(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Flatten()(x)
    output = layers.Dense(1, activation='linear')(x)

    model = imbal.regression.Model(inputs=inputs, outputs=output)

    model.compile(
        loss="mse",
        optimizer=keras.optimizers.Adam(learning_rate=4e-4),
        metrics=["mse"]
    )
    
    model.fit(
        x_train,
        y_train,
        stratify_batches=True,
        validation_split=0.2,
        batch_size=512,
        epochs=10000,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
        ]
    )
```

<div style="display: flex; max-width: 100%;">
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:50%;">
<p>Prediction Plot</p>
<img alt="test"
src="../../_static/regression/image_fit_comparison/regression-true-pred--ae-False-rep-2.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:50%;">
<p>TSNE Visualization</p>
<img alt="test 2"
src="../../_static/regression/image_fit_comparison/tsne_visualization--ae-False-rep-2.png"/>
</div>
</div>

### Balanced Fit

```python
    # Assume the data loading, model construction and compilation as shown in regular fit above.

    kde_bandwidth = imbal.regression.fit_kde(
        y_train,
        bin_count=64
    )
    densities = imbal.regression.get_sample_densities(
        y_train,
        kde_bandwidth
    )

    model.balanced_fit(
        x_train,
        y_train,
        stratify_batches=True,
        sample_density=densities,
        validation_split=0.2,
        batch_size=512,
        epochs=10000,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
        ]
    )
```

<div style="display: flex; max-width: 100%;">
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:50%;">
<p>Prediction Plot</p>
<img alt="test"
src="../../_static/regression/image_fit_comparison/regression-true-pred-balanced-ae-False-rep-2.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:50%;">
<p>TSNE Visualization</p>
<img alt="test 2"
src="../../_static/regression/image_fit_comparison/tsne_visualization-balanced-ae-False-rep-2.png"/>
</div>
</div>

### rRT Fit / Decoupled Fit (representation layer = -2)

```python
    # Assume the data loading and model construction as shown in regular fit above
    model.compile(
        loss="mse",
        optimizer=keras.optimizers.Adam(learning_rate=4e-4),
        metrics=["mse"],
        representation_layer_index=-2
    )
    
    kde_bandwidth = imbal.regression.fit_kde(
        y_train,
        bin_count=64
    )
    densities = imbal.regression.get_sample_densities(
        y_train,
        kde_bandwidth
    )
    
    model.override_second_stage_fit_parameters(
        epochs=10000,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
        ]
    )

    model.rRT_fit(
        x_train,
        y_train,
        stratify_batches=True,
        sample_density=densities,
        validation_split=0.2,
        batch_size=512,
        epochs=10000,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
        ]
    )
```

<div style="display: flex; max-width: 100%;">
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:50%;">
<p>Prediction Plot</p>
<img alt="test"
src="../../_static/regression/image_fit_comparison/regression-true-pred-decoupled-ae-False-rep-2.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:50%;">
<p>TSNE Visualization</p>
<img alt="test 2"
src="../../_static/regression/image_fit_comparison/tsne_visualization-decoupled-ae-False-rep-2.png"/>
</div>
</div>

### rRT Fit / Decoupled Fit (representation layer = -3)

```python
    # Assume the data loading and model construction as shown in regular fit above
    model.compile(
        loss="mse",
        optimizer=keras.optimizers.Adam(learning_rate=4e-4),
        metrics=["mse"],
        representation_layer_index=-3
    )
    
    kde_bandwidth = imbal.regression.fit_kde(
        y_train,
        bin_count=64
    )
    densities = imbal.regression.get_sample_densities(
        y_train,
        kde_bandwidth
    )
    
    model.override_second_stage_fit_parameters(
        epochs=10000,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
        ]
    )

    model.rRT_fit(
        x_train,
        y_train,
        stratify_batches=True,
        sample_density=densities,
        validation_split=0.2,
        batch_size=512,
        epochs=10000,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
        ]
    )
```

<div style="display: flex; max-width: 100%;">
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:50%;">
<p>Prediction Plot</p>
<img alt="test"
src="../../_static/regression/image_fit_comparison/regression-true-pred-decoupled-ae-False-rep-4.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:50%;">
<p>TSNE Visualization</p>
<img alt="test 2"
src="../../_static/regression/image_fit_comparison/tsne_visualization-decoupled-ae-False-rep-4.png"/>
</div>
</div>

### Comparison of Performance

For the table below, frequent samples refer to samples whose label
$18 < y < 80$, and rare samples refer to samples whose label $y \le 18$ or $y \ge 80$.

| Method                            | Epochs  | Time (s) | Frequent MSE | Rare MSE  |
|-----------------------------------|---------|----------|--------------|-----------|
| Regular                           | $38$    | $20.12$  | $177.823$    | $545.133$ |
| Balanced                          | $43$    | $23.08$  | $245.702$    | $501.761$ |
| rRT (representation layer = $-2$) | $46/41$ | $38.40$  | $228.066$    | $504.613$ |
| rRT (representation layer = $-3$) | $39/31$ | $32.91$  | $269.776$    | $442.196$ |

See also: [Comparison of Autoencoder Methods](comparison_of_ae_methods_image.md)
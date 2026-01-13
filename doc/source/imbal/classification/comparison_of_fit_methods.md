# Comparison of Methods: Regular vs. Balanced vs. Decoupled Fit

Below is a comparison of [decoupled fit](decoupled_fit.md) and [balanced fit](balanced_fit.md)
to a standard, unbalanced fit, and a weight-balanced
fit of a subset of the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), picking on the dogs and airplane classes,
in two different data imbalance scenarios.

Each comparison below has three plots, showing a confusion matrix, TSNE
visualization of the latent space of the trained model for the example, and
a plot of the ROC curve for that model's predictions on test data. At the
end of each scenario is a comparison of the AUC and F1 score for the
rare class in the data imbalance. F1 scores are calculated using a
threshold of $0.5$.

## 1:24 Data Imbalance

### Regular Fit

```python
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=2e-5),
        metrics=["accuracy", 'F1Score', auc]
    )
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs
    )
```
#### Regular Fit Results

<div style="display: flex; max-width: 100%;">
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>Confusion Matrix</p>
<img alt="test"
src="../../_static/classification/decoupled_fit/confusion-matrix--low.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>TSNE Visualization</p>
<img alt="test 2"
src="../../_static/classification/decoupled_fit/tsne_visualization--low.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>ROC Curve</p>
<img alt="test 3"
src="../../_static/classification/decoupled_fit/roc-curve--low.png"/>
</div>  
</div>

### Balanced Fit

```python
    parameters = imbal.classification.wrap_model_compile_parameters(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=2e-5),
        metrics=["accuracy", 'F1Score', auc]
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

#### Balanced Fit Results

<div style="display: flex; max-width: 100%;">
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>Confusion Matrix</p>
<img alt="test"
src="../../_static/classification/decoupled_fit/confusion-matrix-balanced-low.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>TSNE Visualization</p>
<img alt="test 2"
src="../../_static/classification/decoupled_fit/tsne_visualization-balanced-low.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>ROC Curve</p>
<img alt="test 3"
src="../../_static/classification/decoupled_fit/roc-curve-balanced-low.png"/>
</div>  
</div>

### Decoupled Fit

```python
    parameters = imbal.classification.wrap_model_compile_parameters(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=2e-5),
        metrics=["accuracy", 'F1Score', auc]
    )

    imbal.classification.decoupled_fit(
        model,
        x_train,
        y_train,
        compile_parameters=parameters,
        epochs=epochs,
        batch_size=batch_size
    )
```

#### Decoupled Fit Results
    
<div style="display: flex; max-width: 100%;">
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>Confusion Matrix</p>
<img alt="test"
src="../../_static/classification/decoupled_fit/confusion-matrix-decoupled-low.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>TSNE Visualization</p>
<img alt="test 2"
src="../../_static/classification/decoupled_fit/tsne_visualization-decoupled-low.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>ROC Curve</p>
<img alt="test 3"
src="../../_static/classification/decoupled_fit/roc-curve-decoupled-low.png"/>
</div>  
</div>

### Comparison of Performance

| Method    | Time (s) | Rare Class F1 Score (threshold=0.5) | AUC      |
|-----------|----------|-------------------------------------|----------|
| Regular   | $9.95$   | $0.0$                               | $0.865$* |
| Balanced  | $13.24$  | $0.240$                             | $0.864$  |
| Decoupled | $14.05$  | $0.211$                             | $0.848$  |

*Some examples have a high AUC, but low F1 score. This is because F1 score
is calculated with a decision threshold of 0.5, while some models can achieve
a near perfect separation of the two classes by using a lower decision
threshold. In these cases, almost all classes are predicted to be of the frequent class,
resulting in an F1 score of nearly 0 for the rare class, while maintaining that a
lower decision threshold exists such that a near perfect separation of classes is achieved,
allowing for a AUROC close to 1.

## 1:120 Data Imbalance

### Regular Fit

```python
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=2e-5),
        metrics=["accuracy", 'F1Score', auc]
    )
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs
    )
```
#### Regular Fit Results

<div style="display: flex; max-width: 100%;">
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>Confusion Matrix</p>
<img alt="test"
src="../../_static/classification/decoupled_fit/confusion-matrix--high.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>TSNE Visualization</p>
<img alt="test 2"
src="../../_static/classification/decoupled_fit/tsne_visualization--high.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>ROC Curve</p>
<img alt="test 3"
src="../../_static/classification/decoupled_fit/roc-curve--high.png"/>
</div>  
</div>

### Balanced Fit

```python
    parameters = imbal.classification.wrap_model_compile_parameters(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=2e-5),
        metrics=["accuracy", 'F1Score', auc]
    )

    imbal.classification.balanced_fit(
        model,
        x_train,
        y_train,high
        compile_parameters=parameters,
        epochs=epochs,
        batch_size=batch_size
    )
```

#### Balanced Fit Results

<div style="display: flex; max-width: 100%;">
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>Confusion Matrix</p>
<img alt="test"
src="../../_static/classification/decoupled_fit/confusion-matrix-balanced-high.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>TSNE Visualization</p>
<img alt="test 2"
src="../../_static/classification/decoupled_fit/tsne_visualization-balanced-high.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>ROC Curve</p>
<img alt="test 3"
src="../../_static/classification/decoupled_fit/roc-curve-balanced-high.png"/>
</div>  
</div>

### Decoupled Fit

```python
    parameters = imbal.classification.wrap_model_compile_parameters(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=2e-5),
        metrics=["accuracy", 'F1Score', auc]
    )

    imbal.classification.decoupled_fit(
        model,
        x_train,
        y_train,
        compile_parameters=parameters,
        epochs=epochs,
        batch_size=batch_size
    )
```

#### Decoupled Fit Results
    
<div style="display: flex; max-width: 100%;">
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>Confusion Matrix</p>
<img alt="test"
src="../../_static/classification/decoupled_fit/confusion-matrix-decoupled-high.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>TSNE Visualization</p>
<img alt="test 2"
src="../../_static/classification/decoupled_fit/tsne_visualization-decoupled-high.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>ROC Curve</p>
<img alt="test 3"
src="../../_static/classification/decoupled_fit/roc-curve-decoupled-high.png"/>
</div>  
</div>

### Comparison of Performance

| Method    | Time (s) | Rare Class F1 Score (threshold=0.5) | AUC      |
|-----------|----------|-------------------------------------|----------|
| Regular   | $9.95$   | $0.0$                               | $0.844$* |
| Balanced  | $13.24$  | $0.092$                             | $0.855$  |
| Decoupled | $14.05$  | $0.079$                             | $0.836$  |

*Some examples have a high AUC, but low F1 score. This is because F1 score
is calculated with a decision threshold of 0.5, while some models can achieve
a near perfect separation of the two classes by using a lower decision
threshold. In these cases, almost all classes are predicted to be of the frequent class,
resulting in an F1 score of nearly 0 for the rare class, while maintaining that a
lower decision threshold exists such that a near perfect separation of classes is achieved,
allowing for a AUROC close to 1.

## Inclusion of Autoencoder (1:24 Imbalance)

### Regular Fit with Autoencoder

<div style="display: flex; max-width: 100%; width:650px">
<img alt="test"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/confusion-matrix-regular-low-ae.png"/>
<img alt="test 2"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/roc-curve-regular-low-ae.png"/>
</div>

### Balanced Fit with Autoencoder

<div style="display: flex; max-width: 100%; width:650px">
<img alt="test"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/confusion-matrix-balanced-low-ae.png"/>
<img alt="test 2"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/roc-curve-balanced-low-ae.png"/>
</div>

### Decoupled Fit with Autoencoder

<div style="display: flex; max-width: 100%; width:650px">
<img alt="test"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/confusion-matrix-decoupled-low-ae.png"/>
<img alt="test 2"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/roc-curve-decoupled-low-ae.png"/>
</div>

### Comparison of Methods

| Method    | Autoencoder? | Epochs    | Time (s) | Rare Class F1 Score (threshold=0.5) | AUC     |
|-----------|--------------|-----------|----------|-------------------------------------|---------|
| Regular   | No           | $30$      | $9.95$   | $0.0$                               | $0.844$ |
| Regular   | Yes          | $600$     | $75.5$   | $0.463$                             | $0.945$ |
| Balanced  | No           | $30$      | $13.24$  | $0.092$                             | $0.855$ |
| Balanced  | Yes          | $600$     | $164.2$  | $0.667$                             | $0.957$ |
| Decoupled | No           | $30/15$   | $14.05$  | $0.079$                             | $0.836$ |
| Decoupled | Yes          | $600/300$ | $161.1$  | $0.713$                             | $0.969$ |
# Comparison of Autoencoder: Regular vs. Balanced vs. cRT Fit on Image Classification

### Regular Fit with Autoencoder (1:24 Imbalance)

<div style="display: flex; max-width: 100%; width:650px">
<img alt="test"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/confusion-matrix-regular-low-ae.png"/>
<img alt="test 2"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/roc-curve-regular-low-ae.png"/>
</div>

### Balanced Fit with Autoencoder (1:24 Imbalance)

<div style="display: flex; max-width: 100%; width:650px">
<img alt="test"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/confusion-matrix-balanced-low-ae.png"/>
<img alt="test 2"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/roc-curve-balanced-low-ae.png"/>
</div>

### cRT Fit with Autoencoder (1:24 Imbalance)

<div style="display: flex; max-width: 100%; width:650px">
<img alt="test"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/confusion-matrix-decoupled-low-ae.png"/>
<img alt="test 2"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/roc-curve-decoupled-low-ae.png"/>
</div>

### Comparison of Methods

| Method   | Autoencoder? | Epochs    | Time (s) | Rare Class F1 Score (threshold=0.5) | AUC     |
|----------|--------------|-----------|----------|-------------------------------------|---------|
| Regular  | No           | $30$      | $9.95$   | $0.0$                               | $0.844$ |
| Regular  | Yes          | $600$     | $75.5$   | $0.463$                             | $0.945$ |
| Balanced | No           | $30$      | $13.24$  | $0.092$                             | $0.855$ |
| Balanced | Yes          | $600$     | $164.2$  | $0.667$                             | $0.957$ |
| cRT      | No           | $30/15$   | $14.05$  | $0.079$                             | $0.836$ |
| cRT      | Yes          | $600/300$ | $161.1$  | $0.713$                             | $0.969$ |

See also: [Comparison of Fit Methods](comparison_of_fit_methods_image.md)
# Comparison of Autoencoder: Regular vs. Balanced vs. Decoupled Fit

### Regular Fit with Autoencoder

<div style="display: flex; max-width: 100%; width:650px">
<img alt="test"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/fit-comparison--ae-True.png"/>
<img alt="test 2"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/tsne_visualization--ae-True.png"/>
</div>

### Balanced Fit with Autoencoder

<div style="display: flex; max-width: 100%; width:650px">
<img alt="test"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/fit-comparison-balanced-ae-True.png"/>
<img alt="test 2"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/tsne_visualization-balanced-ae-True.png"/>
</div>

### Decoupled Fit with Autoencoder

<div style="display: flex; max-width: 100%; width:650px">
<img alt="test"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/fit-comparison-decoupled-ae-True.png"/>
<img alt="test 2"
style="flex: 1; max-width: 50%;"
src="../../_static/classification/decoupled_fit/tsne_visualization-decoupled-ae-True.png"/>
</div>

### Comparison of Methods

| Method    | Autoencoder? | Time (s) | Frequent Sample MSE | Rare Sample MSE |
|-----------|--------------|----------|---------------------|-----------------|
| Regular   | No           | $???$    | $7.5761$            | $0.6142$        |
| Regular   | Yes          | $???$    | $3.5838$            | $1.0461$        |
| Balanced  | No           | $???$    | $6.2100$            | $1.0090$        |
| Balanced  | Yes          | $???$    | $1.9362$            | $1.0344$        |
| Decoupled | No           | $???$    | $1.4113$            | $0.9988$        |
| Decoupled | Yes          | $???$    | $5.2932$            | $0.7171$        |

See also: [Comparison of Fit Methods](comparison_of_fit_methods_tabular.md)
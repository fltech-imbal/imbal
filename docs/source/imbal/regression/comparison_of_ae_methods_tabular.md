# Comparison of Autoencoder: Regular vs. Balanced vs. rRT Fit on Tabular Data

### Regular Fit with Autoencoder

<div style="display: flex; max-width: 100%; width:650px">
<img alt="test"
style="flex: 1; max-width: 50%;"
src="../../_static/regression/decoupled_fit/fit-comparison--ae-True.png"/>
<img alt="test 2"
style="flex: 1; max-width: 50%;"
src="../../_static/regression/decoupled_fit/tsne_visualization--ae-True.png"/>
</div>

### Balanced Fit with Autoencoder

<div style="display: flex; max-width: 100%; width:650px">
<img alt="test"
style="flex: 1; max-width: 50%;"
src="../../_static/regression/decoupled_fit/fit-comparison-balanced-ae-True.png"/>
<img alt="test 2"
style="flex: 1; max-width: 50%;"
src="../../_static/regression/decoupled_fit/tsne_visualization-balanced-ae-True.png"/>
</div>

### rRT Fit with Autoencoder

<div style="display: flex; max-width: 100%; width:650px">
<img alt="test"
style="flex: 1; max-width: 50%;"
src="../../_static/regression/decoupled_fit/fit-comparison-decoupled-ae-True.png"/>
<img alt="test 2"
style="flex: 1; max-width: 50%;"
src="../../_static/regression/decoupled_fit/tsne_visualization-decoupled-ae-True.png"/>
</div>

### Comparison of Methods

| Method   | Autoencoder? | Time (s) | Frequent Sample MSE | Rare Sample MSE |
|----------|--------------|----------|---------------------|-----------------|
| Regular  | No           | $69.2$   | $0.0052$            | $0.1797$        |
| Regular  | Yes          | $92.2$   | $0.0081$            | $0.1968$        |
| Balanced | No           | $100.1$  | $0.0170$            | $0.0581$        |
| Balanced | Yes          | $143.6$  | $0.0212$            | $0.0456$        |
| rRT      | No           | $117.2$  | $0.0077$            | $0.0789$        |
| rRT      | Yes          | $120.0$  | $0.0124$            | $0.1537$        |

See also: [Comparison of Fit Methods](comparison_of_fit_methods_tabular.md)
# Comparison of Autoencoder: Regular vs. Balanced vs. cRT Fit on Tabular Data

### Regular Fit with Autoencoder

<div style="display: flex; max-width: 100%;">
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>Confusion Matrix</p>
<img alt="test"
src="../../_static/classification/decoupled_fit/sep-c/confusion-matrix--ae-True.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>TSNE Visualization</p>
<img alt="test 2"
src="../../_static/classification/decoupled_fit/sep-c/tsne_visualization--ae-True.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>ROC Curve</p>
<img alt="test 3"
src="../../_static/classification/decoupled_fit/sep-c/roc-curve--ae-True.png"/>
</div>  
</div>

### Balanced Fit with Autoencoder

<div style="display: flex; max-width: 100%;">
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>Confusion Matrix</p>
<img alt="test"
src="../../_static/classification/decoupled_fit/sep-c/confusion-matrix-balanced-ae-True.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>TSNE Visualization</p>
<img alt="test 2"
src="../../_static/classification/decoupled_fit/sep-c/tsne_visualization-balanced-ae-True.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>ROC Curve</p>
<img alt="test 3"
src="../../_static/classification/decoupled_fit/sep-c/roc-curve-balanced-ae-True.png"/>
</div>  
</div>

### cRT Fit with Autoencoder

<div style="display: flex; max-width: 100%;">
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>Confusion Matrix</p>
<img alt="test"
src="../../_static/classification/decoupled_fit/sep-c/confusion-matrix-decoupled-ae-True.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>TSNE Visualization</p>
<img alt="test 2"
src="../../_static/classification/decoupled_fit/sep-c/tsne_visualization-decoupled-ae-True.png"/>
</div>
<div style="display:flex; flex-direction:column; flex:1; align-items:center; max-width:33%;">
<p>ROC Curve</p>
<img alt="test 3"
src="../../_static/classification/decoupled_fit/sep-c/roc-curve-decoupled-ae-True.png"/>
</div>  
</div>

### Comparison of Methods

| Method   | Autoencoder? | Time (s) | Rare Class F1 Score (threshold=0.5) | AUC     |
|----------|--------------|----------|-------------------------------------|---------|
| Regular  | No           | $38.36$  | $0.0$                               | $0.883$ |
| Regular  | Yes          | $49.71$  | $0.0$                               | $0.094$ |
| Balanced | No           | $41.34$  | $0.500$                             | $0.537$ |
| Balanced | Yes          | $51.31$  | $0.625$                             | $0.832$ |
| cRT      | No           | $57.93$  | $0.625$                             | $0.858$ |
| cRT      | Yes          | $72.42$  | $0.0$                               | $0.866$ |

See also: [Comparison of Fit Methods](comparison_of_fit_methods_tabular.md)
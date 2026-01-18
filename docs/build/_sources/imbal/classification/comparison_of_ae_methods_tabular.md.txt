# Comparison of Autoencoder: Regular vs. Balanced vs. Decoupled Fit on Tabular Data

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

### Decoupled Fit with Autoencoder

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

| Method    | Autoencoder? | Time (s) | Rare Class F1 Score (threshold=0.5)  | AUC     |
|-----------|--------------|----------|--------------------------------------|---------|
| Regular   | No           | $60.96$  | $0.0640$                             | $0.844$ |
| Regular   | Yes          | $71.71$  | $0.0480$                             | $0.127$ |
| Balanced  | No           | $66.11$  | $0.0635$                             | $0.739$ |
| Balanced  | Yes          | $69.06$  | $0.064$                              | $0.795$ |
| Decoupled | No           | $91.34$  | $0.0635$                             | $0.875$ |
| Decoupled | Yes          | $106.27$ | $0.0711$                             | $0.843$ |

See also: [Comparison of Fit Methods](comparison_of_fit_methods_tabular.md)
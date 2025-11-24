# decoupled_fit

```{eval-rst}
.. autoclass:: imbal.classification.decoupled_fit
```

## Comparison of Standard TensorFlow `fit` vs `decoupled_fit`

| Method    | Time (s) | F1 Score | AUC    |
|-----------|----------|----------|--------|
| Regular   | 9.36     | 0.333    | 0.457  |
| Decoupled | 15.62    | 0.985    | 1.0000 |

### Standard Method Confusion Matrix

<img src="../../_static/classification/decoupled_fit/confusion-matrix-.png" width="450"/>

### Decoupled Fit Confusion Matrix

<img src="../../_static/classification/decoupled_fit/confusion-matrix-decoupled.png" width="450"/>
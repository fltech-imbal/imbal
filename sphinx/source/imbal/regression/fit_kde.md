# fit_kde

```{eval-rst}
.. autoclass:: imbal.regression.fit_kde
```

## Comparison of Methods in One Dimension

Below is a comparison of the different bandwidth fitting methods
available through `fit_kde` across three datasets.

### Dataset 1 (44,485 data points)

<img alt="A series of plots showing the bandwidth comparison for the first dataset" src="/_static/regression/get_densities/bandwidth_method_comparison-dataset_1.png"
style="width:100%"
/>

| Method          | Time (s) |
|-----------------|----------|
| `kl_divergence` | $10.581$ |
| `scott`         | $0.003$  |
| `silverman`     | $0.003$  |

### Dataset 2 (1,531 data points)

<img alt="A series of plots showing the bandwidth comparison for the second dataset" src="/_static/regression/get_densities/bandwidth_method_comparison-dataset_2.png"
style="width:100%"
/>

| Method          | Time (s) |
|-----------------|----------|
| `kl_divergence` | $0.423$  |
| `scott`         | $<0.001$ |
| `silverman`     | $<0.001$ |

### Dataset 3 (16,720 data points)

<img alt="A series of plots showing the bandwidth comparison for the third dataset" src="/_static/regression/get_densities/bandwidth_method_comparison-dataset_3.png"
style="width:100%"
/>

| Method          | Time (s) |
|-----------------|----------|
| `kl_divergence` | $3.122$  |
| `scott`         | $0.002$  |
| `silverman`     | $0.002$  |
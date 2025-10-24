# fit_kde

```{eval-rst}
.. autoclass:: imbal.regression.fit_kde
```

## Comparison of Methods in One Dimension

Below is a comparison of the different bandwidth fitting methods
available through `fit_kde` across three datasets. Note that the
`scott` and `silverman` methods are explicit, "rule of thumb" methods
for finding the bandwidth that take $O(n)$ time. The `kl_divergence`
method is an iterative method that takes $O(kn)$ time, where $k$ is
the number of searches performed per iteration, times the number of
iterations.

### Dataset 1 (44,485 data points)

![A series of plots showing the bandwidth comparison for the first dataset](images/bandwidth_method_comparison-dataset_1.png)

### Dataset 2 (1,531 data points)

![A series of plots showing the bandwidth comparison for the second dataset](images/bandwidth_method_comparison-dataset_2.png)

### Dataset 3 (16,720 data points)

![A series of plots showing the bandwidth comparison for the third dataset](images/bandwidth_method_comparison-dataset_3.png)

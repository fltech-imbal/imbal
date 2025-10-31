# get_densities

```{eval-rst}
.. autoclass:: imbal.regression.get_densities
```

## Comparison of Methods in One Dimension

The interpolations and absolute tolerance estimation methods
exist for the `generate_weights`, which can decrease the
amount of computation time required to compute density values, at 
the cost of a small amount of error in these values. Below is a
comparison of these estimation techniques with the regular
full KDE computation across three different datasets.

### Datasets

For these comparisons, we have made use of three datasets, shown below. These
datasets capture several scenarios for imbalanced data.

#### Dataset 1 (44,485 data points)

<img alt="A histogram showing the data distribution for the first dataset" src="/_static/regression/get_densities/dataset-1.png"
style="width: 450px"
/>

| KDE Optimization Method         | Compute Time (sec) | MAE            | MAPE      |
|---------------------------------|--------------------|----------------|-----------|
| Regular                         | 7.41               | n/a            | n/a       |
| Linear Approximation (320 bins) | 0.25               | $1.88*10^{-4}$ | $0.152\%$ |
| Local Approximation (atol=1e-4) | 3.67               | $1.46*10^{-6}$ | ~$0\%$    |

#### Dataset 2 (1,531 data points)

<img alt="A histogram showing the data distribution for the second dataset" src="/_static/regression/get_densities/dataset-2.png"
style="width: 450px"
/>

| KDE Optimization Method         | Compute Time (sec) | MAE             | MAPE     |
|---------------------------------|--------------------|-----------------|----------|
| Regular                         | 0.018              | n/a             | n/a      |
| Linear Approximation (320 bins) | 0.0029             | $2.00*10^{-2}$  | $0.55\%$ |
| Local Approximation (atol=1e-4) | 0.012              | $5.88*10^{-10}$ | ~$0\%$   |

#### Dataset 3 (16,720 data points)

<img alt="A histogram showing the data distribution for the third dataset" src="/_static/regression/get_densities/dataset-3.png"
style="width: 450px"
/>

| KDE Optimization Method         | Compute Time (sec) | MAE            | MAPE      |
|---------------------------------|--------------------|----------------|-----------|
| Regular                         | 1.22               | n/a            | n/a       |
| Linear Approximation (320 bins) | 0.064              | $4.81*10^{-3}$ | $0.156\%$ |
| Local Approximation (atol=1e-4) | 1.12               | $5.84*10^{-7}$ | ~$0\%$    |

### Argument for Improved 1D Case

For data in one-dimension, we have manually implemented local approximation with absolute tolerance
in a way that bypasses the `scikit-learn` `KernelDensity` object, avoiding some of the overhead
it introduces to work in a general n-dimensional case. Our implementation results in significant
time improvements across all datasets, as shown below. All tests in the table shown below were
performed with an `atol` value of `1e-4`, with no interpolation.

#### Dataset 1

| Method          | Compute Time (sec) | MAE              | MAPE      |
|-----------------|--------------------|------------------|-----------|
| `scitkit-learn` | 23.0               | $2.71*10^{-5}$   | $0.003\%$ |
| `imbal`         | 3.67               | $1.46*10^{-6}$   | ~$0\%$    |

#### Dataset 2

| Method          | Compute Time (sec) | MAE             | MAPE   |
|-----------------|--------------------|-----------------|--------|
| `scitkit-learn` | 0.057              | $1.70*10^{-5}$  | ~$0\%$ |
| `imbal`         | 0.012              | $5.88*10^{-10}$ | ~$0\%$ |

#### Dataset 3

| Method          | Compute Time (sec) | MAE            | MAPE      |
|-----------------|--------------------|----------------|-----------|
| `scitkit-learn` | 6.21               | $2.25*10^{-5}$ | $0.003\%$ |
| `imbal`         | 1.124*10^{-7}$ | ~$0\%$    |

### Comparison of Methods in Two Dimensions

For testing of the different estimation methods in two dimensions, we made use of a
"toy" 2D gaussian dataset with a mean of $0$ and a covariance of $\begin{bmatrix} 1 & 0 \\ 0 & 0.15 \end{bmatrix}$.

#### True Distribution (left) vs KDE (right)

<img alt="A histogram showing the data distribution for the 2d toy dataset" src="/_static/regression/get_densities/multidim-kde-comparison.png"
style="width:100%"
/>

| Method                             | Time (s) | MAE            | MAPE     |
|------------------------------------|----------|----------------|----------|
| Regular                            | 3.41     | n/a            | n/a      |
| Local Approximation (atol=1e-4)    | 0.78     | $3.83*10^{-5}$ | $0.04\%$ |
| Linear Interpolation (64 bins/dim) | 0.42     | $2.21*10^{-3}$ | $1.43\%$ |
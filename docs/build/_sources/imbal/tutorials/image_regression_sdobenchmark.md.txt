# Image regression on SDOBenchmark

The purpose of these tutorials is to demonstrate the utility
of the `imbal` package for the application of regression on
image data.

The dataset used in these tutorials is a subset of the
[SDOBenchmark dataset](https://i4ds.github.io/SDOBenchmark/).
The subset consists of 500 training samples and 100 test samples,
all of which contain the 10 images from the timestamp 10 minute
prior to the prediction time. The full subset of data
used in these examples can be found in the `tutorials/data/SDOBenchmark`
folder in the `imbal` repository.

All the code shown in the provided tutorials can be found in
the `tutorials/SDO` folder in the `imbal` repository.

## Tutorials

Below is a table containing links to tutorials for possible model fit approaches that can be achieved through `imbal`.
Each column corresponds to a type of fit that can be performed with `imbal.regression` (regular, balanced, rRT).
Each row corresponds to some additional modification to the approach:
- Basic: The basic approach, with no additional modification
- With AE: Includes automatic autoencoder branch generation
- With Val: Includes splitting of validation data, used to reduce overfitting and allows for comparison between the performance of multiple sample  weight candidates

|                 | Regular Fit                                      | Balanced Fit                                       | rRT Fit                                  |
|-----------------|--------------------------------------------------|----------------------------------------------------|------------------------------------------|
| Basic           | [Regular](SDO/regression/regular_fit.md)         | [Balanced](SDO/regression/balanced_fit.md)         | [rRT](SDO/regression/rrt_fit.md)         |
| With AE         | [Regular+AE](SDO/regression/regular_fit_ae.md)   | [Balanced+AE](SDO/regression/balanced_fit_ae.md)   | [rRT+AE](SDO/regression/rrt_fit_ae.md)   |
| With Validation | [Regular+val](SDO/regression/regular_fit_val.md) | [Balanced+val](SDO/regression/balanced_fit_val.md) | [rRT+val](SDO/regression/rrt_fit_val.md) |

- [Using `imbal.metrics` in model regression](SDO/regression/metrics.md)
- [Model representation visualization using t-SNE](SDO/regression/visualization.md)
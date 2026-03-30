# Image regression on SDOBenchmark

The purpose of these tutorials is to demonstrate the utility
of the `imbal` package for the application of regression on
image data.

The dataset used in these tutorials is a subset of the
[SDOBenchmark dataset](https://i4ds.github.io/SDOBenchmark/).
The subset consists of 1000 training samples and 300 test samples,
all of which contain the 10 images from the timestamp 10 minute
prior to the prediction time. The full subset of data
used in these examples can be found in the `tutorials/data/SDOBenchmark`
folder in the `imbal` repository.

All the code shown in the provided tutorials can be found in
the `tutorials/SDO` folder in the `imbal` repository.

## Tutorials:
- [Regular Regression on SDOBenchmark](SDO/regular_fit.md)
- [Balanced Regression on SDOBenchmark](SDO/balanced_fit.md)
- [rRT Regression on SDOBenchmark](SDO/rrt_fit.md)
import numpy as np
import imbal

labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2])
kde_bandwidth = imbal.regression.fit_kde(labels, bin_count=3)
densities = imbal.regression.get_densities(labels, kde_bandwidth)

print(densities)

linear_interpolation_densities = imbal.regression.get_densities(
    labels,
    kde_bandwidth,
    interpolation_method='linear',
    interpolation_samples=5
)

print(linear_interpolation_densities)

local_approx_densities = imbal.regression.get_densities(
    labels,
    kde_bandwidth,
    atol = 0.1
)

# For this example dataset, there are no errors, even for higher tolerance values
print(local_approx_densities)
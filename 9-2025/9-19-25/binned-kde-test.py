from imbal.sample_weighting import generate_regression_weights
import numpy as np

labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2]).reshape(-1, 1)

generate_regression_weights(labels, bandwidth='binned')
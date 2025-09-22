from imbal.regression import generate_weights
import numpy as np

labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2]).reshape(-1, 1)

generate_weights(labels, bandwidth='binned')
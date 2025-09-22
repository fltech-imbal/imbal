from imbal.util.sample_weighting import generate_sample_weights
import numpy as np

labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2])
# np.random.shuffle(labels)

print(generate_sample_weights(labels, {
    0: 1/6,
    1: 1/3,
    2: 1/2
}).tolist())

print(generate_sample_weights(labels).tolist())

print(generate_sample_weights(labels, mode='regression').tolist())
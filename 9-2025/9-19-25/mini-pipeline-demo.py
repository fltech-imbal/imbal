import imbal
import numpy as np

data = np.arange(20).reshape(-1, 1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2]).reshape(-1, 1)
weights = imbal.sample_weighting.generate_classification_weights(labels)
print(weights)

batches = imbal.stratified_sampling.DatasetWithBatching(data, labels, weights, num_batches=5)

for batch in batches:
    data, labels, weights = batch
    niceify = lambda x: x.numpy().reshape(-1,).tolist()
    data, labels, weights = niceify(data), niceify(labels), niceify(weights)
    print(data)
    print(labels)
    print(weights)
    print()

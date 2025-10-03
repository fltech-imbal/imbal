import imbal.classification as im
import numpy as np

data = np.arange(20).reshape(-1, 1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2]).reshape(-1, 1)
weights = im.generate_weights(labels)
print(weights)

batches = im.DatasetWithBatching(data, labels, weights, num_batches=5)

for batch in batches:
    data, labels, weights = batch
    niceify = lambda x: x.numpy().reshape(-1,).tolist()
    data, labels, weights = niceify(data), niceify(labels), niceify(weights)
    print(data)
    print(labels)
    print(weights)
    print()

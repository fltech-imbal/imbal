from imbal.sampling import StratifiedSampler
from keras.datasets import mnist
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

sampler = StratifiedSampler(
    x_train,
    y_train,
    batch_size=40,
    class_weights={}
)

print(tf.unique_with_counts(y_train[:100]))

print("Building stratified samples...")
sampler.build()
print("Done!")

cumulative_sum = 0
for i in range(len(sampler)):
    sample = sampler[i][1:]
    if i < 8:
        print(np.transpose(sample, axes=[0, 2, 1]))
        print(sum(np.reshape(sample[1].numpy(), (-1,))))
    cumulative_sum += sum(np.reshape(sample[1].numpy(), (-1,)))


print(cumulative_sum)





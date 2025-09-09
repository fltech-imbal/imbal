from imbal.sampling import StratifiedSampler
from keras.datasets import mnist
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

sampler = StratifiedSampler(
    x_train[:1000],
    y_train[:1000],
    batch_size=40,
    # class_weights={}
)

print(tf.unique_with_counts(y_train[:1000])[0])
print(tf.unique_with_counts(y_train[:1000])[2])

trigger = 0
cumulative_sum = 0
for i in range(len(sampler)):
    sample = sampler[i][1:]

    if i < 8:
        print(np.astype(np.transpose(sample, axes=[0, 2, 1])[0][0], np.int32))
        print(np.transpose(sample, axes=[0, 2, 1])[1][0])
        print(sum(np.reshape(sample[1].numpy(), (-1,))))
    if i == 8:
        print('\n...\n')

    if sum(np.reshape(sample[1].numpy(), (-1,))) >= 0.04 and trigger < 2:
        print(np.astype(np.transpose(sample, axes=[0, 2, 1])[0][0], np.int32))
        print(np.transpose(sample, axes=[0, 2, 1])[1][0])
        print(sum(np.reshape(sample[1].numpy(), (-1,))))
        trigger += 1

    cumulative_sum += sum(np.reshape(sample[1].numpy(), (-1,)))


print(cumulative_sum)





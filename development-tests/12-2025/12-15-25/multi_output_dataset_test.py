import imbal
import keras
import numpy as np
from keras.utils import to_categorical

"""
Load data
"""

num_classes = 10
input_shape = (32, 32, 3)

DATASET_PERCENTAGE = 0.8
TRAIN_SPLIT = 0.8

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_combined = np.concatenate((x_train, x_test), axis=0)
y_combined = np.concatenate((y_train, y_test), axis=0)
y_combined = y_combined.reshape(-1,)

print(x_combined.shape)
print(y_combined.shape)

num_data = x_combined.shape[0]
percent_index = int(num_data * DATASET_PERCENTAGE)
num_data = x_combined.shape[0]
split_index = int(num_data * TRAIN_SPLIT)
x_train, x_test = x_combined[:split_index], x_combined[split_index:]
y_train, y_test = y_combined[:split_index], y_combined[split_index:]
print('x_train', x_train.shape)
print('y_train',y_train.shape)
print('x_test',x_test.shape)
print('y_test',y_test.shape)

class_split = []
for i in range(num_classes):
    class_split.append(len(y_train[y_train == i]))
print(class_split)

x_train_filter = []
y_train_filter = []
x_test_filter = []
y_test_filter = []
for i in range(num_classes):
        if i == 6:
            break
        if i == 0:
            x_train_filter.append(x_train[y_train == i])
            y_train_filter.append(y_train[y_train == i])
            x_test_filter.append(x_test[y_test == i])
            y_test_filter.append(y_test[y_test == i])
        else:
            if i < 5:
                continue
            x_train_filter.append(x_train[y_train == i][:200])
            y_train_filter.append(np.ones(y_train[y_train == i][:200].shape))
            x_test_filter.append(x_test[y_test == i][:50])
            y_test_filter.append(np.ones(y_test[y_test == i][:50].shape))

x_train = np.concatenate(x_train_filter)
x_test = np.concatenate(x_test_filter)
y_train = np.concatenate(y_train_filter)
y_test = np.concatenate(y_test_filter)

train_shuffle = np.random.permutation(x_train.shape[0])
test_shuffle = np.random.permutation(x_test.shape[0])
x_train = x_train[train_shuffle]
x_test = x_test[test_shuffle]
y_train = y_train[train_shuffle]
y_test = y_test[test_shuffle]

x_train = x_train / 255
x_test = x_test / 255

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

sample_weights = imbal.classification.generate_sample_weights(y_train)

# x_train = np.array([1, 2, 3]).reshape(-1, 1)
# y_train = np.array([0, 0, 1]).reshape(-1, 1)
# sample_weights = imbal.classification.generate_sample_weights(y_train)

# x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
# y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1)
# sample_weights = imbal.classification.generate_sample_weights(y_train)

dataset = imbal.classification.DatasetWithBatching(
    x_train,
    [y_train, x_train],
    sample_weights=sample_weights,
    multi_output=True,
    batch_size=2,
)


import re
for i in range(1):
    x, y, weights = dataset[i]
    print(i)
    print("\t" + 'X')
    print("\t" + re.sub('\s+', ' ', str(x)))
    print("\t" + 'Y')
    print("\t" + re.sub('\s+', ' ', str(y)))
    print("\t" + 'WEIGHTS')
    print("\t" + re.sub('\s+', ' ', str(weights)))
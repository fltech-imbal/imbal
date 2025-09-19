import numpy as np

from imbal.stratified_sampling import DatasetWithBatching

def test_sampler(s, text) -> None:
    print(f'{text}\n' + '=' * 80)
    for i in range(2):
        labels = []
        weights = []
        batches = []
        batch_sums = []
        weight_sum = 0
        for i in range(len(s)):
            sample = s[i]
            labels.append(sample[1].numpy().reshape((-1,)).astype(int).tolist())
            weights.append(np.round(sample[2].numpy().reshape((-1,)), decimals=3).tolist())
            batches.append(sample[0].numpy().reshape((-1,)).astype(int).tolist())
            batch_sums.append(round(sum(sample[2].numpy().reshape((-1,)).tolist()), 3))
            weight_sum += sum(sample[2].numpy().reshape((-1,)))

        print('Batches:')
        print(batches)
        print('Labels:')
        print(labels)
        print('Weights:')
        print(weights)
        print('Weight sums per batch:')
        print(batch_sums)
        print('Total Weights:')
        print(weight_sum)
        print()
        print('=' * 80)
        s.on_epoch_end()
        print()


# default weighting, 9 "cats", 1 "dragon"
data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]).reshape(-1,1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(-1,1)

sampler_1 = DatasetWithBatching(data, labels, num_batches=2)
test_sampler(sampler_1, 'TEST 1')

# default weighting, 7 "cats", 3 "dragons"
data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]).reshape(-1,1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1]).reshape(-1,1)

sampler_2 = DatasetWithBatching(data, labels, num_batches=2)
test_sampler(sampler_2, 'TEST 2')

# instance weighting, 9 "cats", 1 "dragon"
data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1,1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(-1,1)
weights = np.choose(labels, [0.5/9, 0.5])

sampler_3 = DatasetWithBatching(data, labels, num_batches=2, sample_weights=weights)
test_sampler(sampler_3, 'TEST 3')

# instance weighting, 7 "cats", 3 "dragons"
data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]).reshape(-1,1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1]).reshape(-1,1)
weights = np.choose(labels, [0.5/7, 0.5/3])

sampler_4 = DatasetWithBatching(data, labels, num_batches=2,sample_weights=weights)
test_sampler(sampler_4, 'TEST 4')

############################################################################

# default weighting, 10 "cats", 9 "dogs", 1 "dragon"
data = np.arange(20).reshape(-1,1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]).reshape(-1,1)

sampler_4 = DatasetWithBatching(data, labels, num_batches=3)
test_sampler(sampler_4, 'COMPLEX TEST 1')

# default weighting, 9 "cats", 9 "dogs", 2 "dragon"
data = np.arange(20).reshape(-1,1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]).reshape(-1,1)

sampler_4 = DatasetWithBatching(data, labels, num_batches=3)
test_sampler(sampler_4, 'COMPLEX TEST 2')

# default weighting, 8 "cats", 8 "dogs", 4 "dragon"
data = np.arange(20).reshape(-1,1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]).reshape(-1,1)

sampler_4 = DatasetWithBatching(data, labels, num_batches=3)
test_sampler(sampler_4, 'COMPLEX TEST 3')

# balanced weighting, 10 "cats", 9 "dogs", 1 "dragon"
data = np.arange(20).reshape(-1,1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]).reshape(-1,1)
weights = np.choose(labels, [0.33/10, 0.33/9, 0.34])

sampler_4 = DatasetWithBatching(data, labels, num_batches=3, sample_weights=weights)
test_sampler(sampler_4, 'COMPLEX TEST 4')

# balanced weighting, 9 "cats", 9 "dogs", 2 "dragon"
data = np.arange(20).reshape(-1,1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]).reshape(-1,1)
weights = np.choose(labels, [0.33/9, 0.33/9, 0.34/2])

sampler_4 = DatasetWithBatching(data, labels, num_batches=3, sample_weights=weights)
test_sampler(sampler_4, 'COMPLEX TEST 5')

# balanced weighting, 8 "cats", 8 "dogs", 4 "dragon"
data = np.arange(20).reshape(-1,1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]).reshape(-1,1)
weights = np.choose(labels, [0.33/8, 0.33/8, 0.34/4])

sampler_4 = DatasetWithBatching(data, labels, num_batches=3, sample_weights=weights)
test_sampler(sampler_4, 'COMPLEX TEST 6')

import imbal
train_set, test_set = imbal.stratified_sampling.split(data, labels, weights, test_size=0.25)
x_train, y_train, w_train = train_set.get_unzipped()
x_test, y_test, w_test = test_set.get_unzipped()

print(np.reshape(x_train, (-1,)))
print(np.reshape(y_train, (-1,)))
print(np.reshape(w_train, (-1,)))
print(np.reshape(x_test, (-1,)))
print(np.reshape(y_test, (-1,)))
print(np.reshape(w_test, (-1,)))
print()

# regression_data_test
data = np.arange(21).reshape(-1,1)
labels = np.arange(21).reshape(-1,1)
weights = (np.ones(21) / 21).reshape(-1, 1)

train_set, test_set = imbal.stratified_sampling.split(data, labels, weights, test_size=0.20, mode='regression')
x_train, y_train, w_train = train_set.get_unzipped()
x_test, y_test, w_test = test_set.get_unzipped()

print(np.reshape(x_train, (-1,)))
print(np.reshape(y_train, (-1,)))
print(np.reshape(w_train, (-1,)))
print(np.reshape(x_test, (-1,)))
print(np.reshape(y_test, (-1,)))
print(np.reshape(w_test, (-1,)))
print()

# reg batching, balanced weighting (also handles scenario where sum(weights) != 1)
data = np.arange(20).reshape(-1,1)
labels = np.arange(20).reshape(-1,1)
weights = np.arange(20).reshape(-1,1)

sampler_4 = DatasetWithBatching(data, labels, num_batches=3, mode='regression', sample_weights=weights)
test_sampler(sampler_4, 'REG TEST 1')


# instance weighting, 7 "cats", 3 "dragons"
data = np.arange(20).reshape(-1,1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2]).reshape(-1,1)

sampler_4 = DatasetWithBatching(data, labels, num_batches=3)
test_sampler(sampler_4, 'TEST 4')







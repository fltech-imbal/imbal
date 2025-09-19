import numpy as np
from keras.src.trainers.data_adapters.data_adapter_utils import class_weight_to_sample_weights
# from sklearn.model_selection import train_test_split

from imbal.stratified_sampling import WeightBalancedSampler

def test_sampler(s) -> None:
    labels = []
    weights = []
    batches = []
    batch_sums = []
    for i in range(len(s)):
        sample = s[i]
        labels.append(sample[1].numpy().reshape((-1,)).astype(int).tolist())
        weights.append(np.round(sample[2].numpy().reshape((-1,)), decimals=3).tolist())
        batches.append(sample[0].numpy().reshape((-1,)).astype(int).tolist())
        batch_sums.append(round(sum(sample[2].numpy().reshape((-1,)).tolist()), 3))

    print('Batches:')
    print(batches)
    print('Labels:')
    print(labels)
    print('Weights:')
    print(weights)
    print('Weight sums per batch:')
    print(batch_sums)
    print('Total Weights:')
    print(sum(batch_sums))
    print()
    print('=' * 80)
    s.on_epoch_end()
    print()


# default weighting, 9 "cats", 1 "dragon"
data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]).reshape(-1,1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(-1,1)

sampler_1 = WeightBalancedSampler(data, labels, num_batches=2, sample_weights={})
print('TEST 1\n'+'='*80)
test_sampler(sampler_1)
test_sampler(sampler_1)
test_sampler(sampler_1)

# default weighting, 7 "cats", 3 "dragons"
data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]).reshape(-1,1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1]).reshape(-1,1)

sampler_2 = WeightBalancedSampler(data, labels, num_batches=2,sample_weights={})
print('TEST 2\n'+'='*80)
test_sampler(sampler_2)
test_sampler(sampler_2)
test_sampler(sampler_2)

# instance weighting, 9 "cats", 1 "dragon"
data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1,1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(-1,1)

sampler_3 = WeightBalancedSampler(data, labels, num_batches=2,
    sample_weights={
        0 : 0.5/9,
        1: 0.5
    }
)
print('TEST 3\n'+'='*80)
test_sampler(sampler_3)
test_sampler(sampler_3)
test_sampler(sampler_3)

# instance weighting, 7 "cats", 3 "dragons"
data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]).reshape(-1,1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1]).reshape(-1,1)

sampler_4 = WeightBalancedSampler(data, labels, num_batches=2,
    sample_weights={
        0 : 0.5/7,
        1: 0.5/3
    }
)
print('TEST 4\n'+'='*80)
test_sampler(sampler_4)
test_sampler(sampler_4)
test_sampler(sampler_4)

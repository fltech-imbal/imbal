import unittest
import numpy as np
from imbal.stratified_sampling import DatasetWithBatching

class TestStratifiedBatcher(unittest.TestCase):
    def test_preserve_weights_on_sparse_data(self) -> None:
        """
        This test presents a scenario where there are 20 data points across
        4 class labels. A stratified batching is being performed on this
        data across four batches.

        Classes 0 and 1 contain a sufficient amount of data points to have
        at least one entry of each class in each batch. Class 2 contains
        three entries, requiring at least one duplicate of the class entries
        to be made to ensure one copy can be contained in each batch. Class
        3 contains 1 entry, requiring three duplicates of the class entry.

        Using weights of 1 for each entry, the expected behavior of the
        stratified batching is that each batch is made with the same amount
        of data, labels, and weights outputted, that each batch is either six
        or seven entries long (10 + 6 + 2*3 + 4*1 = 22, 22/4 = 6.5), that
        data labels stay with their corresponding data points, even after
        shuffling, and that weights stay with their corresponding data points
        while also being adjusted based on duplications (1, 1, 0.5, and 0.25,
        for each class respectively). And of course, each batch should
        contain at least one of each entry.
        """
        data = np.arange(20).reshape(-1, 1)
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3]).reshape(-1, 1)
        weights = np.ones(20).reshape(-1, 1)

        batcher = DatasetWithBatching(data, labels, num_batches=4, sample_weights=weights)
        for i in range(len(batcher)):
            batch = batcher[i]

            batch_data, batch_labels, batch_weights = batch
            batch_data, batch_labels, batch_weights = batch_data.numpy(), batch_labels.numpy(), batch_weights.numpy()

            self.assertTrue(batch_data.shape[0] == batch_labels.shape[0])
            self.assertTrue(batch_data.shape[0] == batch_weights.shape[0])
            self.assertTrue(batch_data.shape[0] == 6 or batch_data.shape[0] == 7)

            fetched_labels = labels[batch_data].reshape(-1, 1)

            self.assertTrue(np.allclose(batch_labels, fetched_labels))
            self.assertTrue(np.allclose(np.choose(batch_labels, [1, 1, 0.5, 0.25]), batch_weights))
            self.assertTrue(np.unique(batch_labels).shape[0] == 4)

    def test_preserve_data_label_weight_association_regression(self) -> None:
        """
        This test presents a scenario where there are 2000 data-label-weight tuples,
        each specified in a column vector of length 2000. A regression stratified
        batch sampling is being performed across this data set.

        Data and labels are completely random values, but the weights of each sample
        correspond to the index of each data point in the list, which is used for test
        verification.

        The tests being performed are as follows:
        - Sample weights are preserved after batching (no division, unlike class-based
          stratified batching
        - Using the preserved weights as indices, the corresponding data, label, and
          weight associations are preserved after batching
        - Across all batches, each sample only appears once. This means:
          - Each data point appears once
          - Each label appears once
          - Each label appears once, which should means that if the concatenated list
            of all batch weights is sorted, it should be equal to np.arange(2000)
        """
        data = np.random.rand(2000).reshape(-1, 1)
        labels = np.random.rand(2000).reshape(-1, 1)
        weights = np.arange(2000).reshape(-1, 1)

        batcher = DatasetWithBatching(data, labels, batch_size=64, mode='reg', sample_weights=weights)

        unique_data = np.array([])
        unique_labels = np.array([])
        unique_weights = np.array([])
        for i in range(len(batcher)):
            batch = batcher[i]
            batch_data , batch_labels, batch_weights = batch
            batch_data, batch_labels, batch_weights = batch_data.numpy(), batch_labels.numpy(), batch_weights.numpy()
            self.assertTrue(np.allclose(batch_weights, batch_weights.astype(int)))
            batch_weights = batch_weights.astype(int)
            fetched_data = data[batch_weights].reshape(-1, 1)
            fetched_labels = labels[batch_weights].reshape(-1, 1)
            self.assertTrue(np.allclose(fetched_data, batch_data))
            self.assertTrue(np.allclose(fetched_labels, batch_labels))

            unique_data = np.concatenate((unique_data, np.unique(batch_data.reshape(-1,))))
            unique_labels = np.concatenate((unique_labels, np.unique(batch_labels.reshape(-1,))))
            unique_weights = np.concatenate((unique_weights, np.unique(batch_weights.reshape(-1,))))
        self.assertTrue(unique_data.shape[0] == 2000)
        self.assertTrue(unique_labels.shape[0] == 2000)
        self.assertTrue(unique_weights.shape[0] == 2000)
        unique_weights = np.sort(unique_weights)
        self.assertTrue(np.allclose(unique_weights.reshape(-1, 1), weights))

    def test_simple_case_1(self) -> None:
        data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1)
        sampler_1 = DatasetWithBatching(data, labels, num_batches=2)
        self.assertTrue(len(sampler_1) == 2)
        self.assertTrue(9 in sampler_1[0][0].numpy().reshape(-1,))
        self.assertTrue(9 in sampler_1[1][0].numpy().reshape(-1,))
        self.assertTrue(1 in sampler_1[0][1].numpy().reshape(-1,))
        self.assertTrue(1 in sampler_1[1][1].numpy().reshape(-1,))

    def test_simple_case_2(self) -> None:
        data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1)
        weights = np.choose(labels, [0.5 / 9, 0.5])
        sampler_2 = DatasetWithBatching(data, labels, num_batches=2, sample_weights=weights)
        self.assertTrue(0.25 in sampler_2[0][2].numpy().reshape(-1, ))
        self.assertTrue(0.25 in sampler_2[1][2].numpy().reshape(-1, ))

    def test_simple_case_3(self) -> None:
        data = np.arange(20).reshape(-1, 1)
        labels = np.arange(20).reshape(-1, 1)
        weights = np.arange(20).reshape(-1, 1)
        sampler_3 = DatasetWithBatching(data, labels, num_batches=3, mode='regression', sample_weights=weights)
        self.assertTrue(np.allclose(sampler_3[0][0], sampler_3[0][1]))
        self.assertTrue(np.allclose(sampler_3[0][0], sampler_3[0][2]))
        self.assertTrue(np.count_nonzero(
            [0 in sampler_3[0][0].numpy(), 1 in sampler_3[0][0].numpy(), 2 in sampler_3[0][0].numpy()]) == 1)
        self.assertTrue(np.count_nonzero(
            [3 in sampler_3[0][0].numpy(), 4 in sampler_3[0][0].numpy(), 5 in sampler_3[0][0].numpy()]) == 1)
        self.assertTrue(np.count_nonzero(
            [6 in sampler_3[0][0].numpy(), 7 in sampler_3[0][0].numpy(), 8 in sampler_3[0][0].numpy()]) == 1)
        self.assertTrue(np.count_nonzero(
            [9 in sampler_3[0][0].numpy(), 10 in sampler_3[0][0].numpy(), 11 in sampler_3[0][0].numpy()]) == 1)
        self.assertTrue(np.count_nonzero(
            [12 in sampler_3[0][0].numpy(), 13 in sampler_3[0][0].numpy(), 14 in sampler_3[0][0].numpy()]) == 1)
        self.assertTrue(np.count_nonzero(
            [15 in sampler_3[0][0].numpy(), 16 in sampler_3[0][0].numpy(), 17 in sampler_3[0][0].numpy()]) == 1)
        self.assertTrue(np.count_nonzero(
            [18 in sampler_3[0][0].numpy(), 19 in sampler_3[0][0].numpy()]) == 1)

if __name__ == '__main__':
    unittest.main()
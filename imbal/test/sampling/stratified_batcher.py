import unittest
import numpy as np
from imbal.sampling import StratifiedBatcher

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

        batcher = StratifiedBatcher(data, labels, num_batches=4, sample_weights=weights)
        for i in range(len(batcher)):
            batch = batcher[i]

            batch_data , batch_labels, batch_weights = batch
            batch_data, batch_labels, batch_weights = batch_data.numpy(), batch_labels.numpy(), batch_weights.numpy()

            self.assertTrue(batch_data.shape[0] == batch_labels.shape[0])
            self.assertTrue(batch_data.shape[0] == batch_weights.shape[0])
            self.assertTrue(batch_data.shape[0] == 6 or batch_data.shape[0] == 7)

            fetched_labels = labels[batch_data].reshape(-1, 1)

            self.assertTrue(np.allclose(batch_labels, fetched_labels))
            self.assertTrue(np.allclose(np.choose(batch_labels, [1, 1, 0.5, 0.25]), batch_weights))

    def test_preserve_data_label_weight_association_regression(self) -> None:
        data = np.random.rand(2000).reshape(-1, 1)
        labels = np.random.rand(2000).reshape(-1, 1)
        weights = np.arange(2000).reshape(-1, 1)

        batcher = StratifiedBatcher(data, labels, batch_size=64, mode='reg', sample_weights=weights)
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


if __name__ == '__main__':
    unittest.main()
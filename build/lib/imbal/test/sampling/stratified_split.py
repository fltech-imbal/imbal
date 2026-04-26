import unittest
import numpy as np
import imbal.classification as imc
import imbal.regression as imr

class TestStratifiedSplit(unittest.TestCase):
    def test_stratified_split_classification_sparse_data(self) -> None:
        """
        This test presents a scenario where a class-based train/test
        split is being performed on a small set of data. The goal is
        to ensure the stratified split is being performed while ensuring
        that data-label-weight associations are preserved

        The data points used in the tests correspond to the index of the
        data point, which can be used during tests to compare the
        resultant splits with the original data.

        The tests being performed are as follows:
        - A proper stratified 75/25 split is being performed. This means
          that the train set will have six instances of classes 0 and 1, and
          three instances of class 2, while the test set will have two
          instances of classes 0 and 1, and one instances of class 2.
        - By comparing with the original data before splitting, that
          data-label-weight associations are preserved.
        """
        data = np.arange(20).reshape(-1, 1)
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]).reshape(-1, 1)
        weights = np.random.rand(20).reshape(-1, 1)

        train_set, test_set = imc.split(data, labels, weights, test_size=0.25)
        x_train, y_train, w_train = train_set.get_unzipped()
        x_test, y_test, w_test = test_set.get_unzipped()

        _, train_unique_counts = np.unique(y_train, return_counts=True)
        _, test_unique_counts = np.unique(y_test, return_counts=True)

        self.assertTrue(np.array_equal(train_unique_counts, (6, 6, 3)))
        self.assertTrue(np.array_equal(test_unique_counts, (2, 2, 1)))
        self.assertTrue(np.array_equal(np.choose(x_train, labels), y_train))
        self.assertTrue(np.array_equal(np.choose(x_test, labels), y_test))
        self.assertTrue(np.array_equal(np.choose(x_train, weights), w_train))
        self.assertTrue(np.array_equal(np.choose(x_test, weights), w_test))

    def test_stratified_split_regression_sparse_data(self) -> None:
        """
        This test presents a scenario where a class-based train/test
        split is being performed on a small set of data. The goal is
        to ensure the stratified split is being performed while ensuring
        that data-label-weight associations are preserved

        The data points used in the tests correspond to the index of the
        data point, which can be used during tests to compare the
        resultant splits with the original data.

        The tests being performed are as follows:
        - A proper stratified 80/20 split is being performed. This means
          that the train set will have six instances of classes 0 and 1, and
          three instances of class 2, while the test set will have two
          instances of classes 0 and 1, and one instances of class 2.
        - By comparing with the original data before splitting, that
          data-label-weight associations are preserved.
        """
        data = np.arange(20)
        labels = np.arange(20)
        weights = (np.ones(20) / 20)

        indices = np.arange(20)
        np.random.shuffle(indices)
        data = data[indices].reshape(-1, 1)
        labels = labels[indices].reshape(-1, 1)
        weights = weights[indices].reshape(-1, 1)

        train_set, test_set = imr.split(data, labels, weights, test_size=0.20)
        x_train, y_train, w_train = train_set.get_unzipped()
        x_test, y_test, w_test = test_set.get_unzipped()

        x_combined = np.concatenate((x_train, x_test))
        y_combined = np.concatenate((y_train, y_test))
        w_combined = np.concatenate((w_train, w_test))

        combined_label_uniques, counts = np.unique(y_combined, return_counts=True)

        self.assertTrue(x_train.shape[0] == 16)
        self.assertTrue(y_train.shape[0] == 16)
        self.assertTrue(w_train.shape[0] == 16)
        self.assertTrue(x_test.shape[0] == 4)
        self.assertTrue(y_test.shape[0] == 4)
        self.assertTrue(w_test.shape[0] == 4)
        self.assertTrue(combined_label_uniques.shape[0] == 20)
        self.assertTrue(np.allclose(counts.reshape(-1,), np.ones(20)))

if __name__ == '__main__':
    unittest.main()
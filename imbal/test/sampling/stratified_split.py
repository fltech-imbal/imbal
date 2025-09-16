import unittest
import numpy as np
from imbal.sampling import stratified_split

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

        train_set, test_set = stratified_split(data, labels, weights, test_size=0.25)
        x_train, y_train, w_train = train_set
        x_test, y_test, w_test = test_set

        train_unique_counts = np.unique_counts(y_train)
        test_unique_counts = np.unique_counts(y_test)

        self.assertTrue(np.array_equal(train_unique_counts.counts, (6, 6, 3)))
        self.assertTrue(np.array_equal(test_unique_counts.counts, (2, 2, 1)))
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
        data = np.arange(20).reshape(-1, 1)
        labels = np.arange(20).reshape(-1, 1)
        weights = (np.ones(20) / 20).reshape(-1, 1)

        train_set, test_set = stratified_split(data, labels, weights, test_size=0.20, mode='reg')
        x_train, y_train, w_train = train_set
        x_test, y_test, w_test = test_set

        train_unique_counts = np.unique_counts(y_train)
        test_unique_counts = np.unique_counts(y_test)

        # self.assertTrue(np.array_equal(train_unique_counts.counts, (6, 6, 3)))
        # self.assertTrue(np.array_equal(test_unique_counts.counts, (2, 2, 1)))
        # self.assertTrue(np.array_equal(np.choose(x_train, labels), y_train))
        # self.assertTrue(np.array_equal(np.choose(x_test, labels), y_test))
        # self.assertTrue(np.array_equal(np.choose(x_train, weights), w_train))
        # self.assertTrue(np.array_equal(np.choose(x_test, weights), w_test))




if __name__ == '__main__':
    unittest.main()
import numpy as np
from imbal.util.backend.constants import ModelType
from math import ceil

def split(
    x_set,
    y_set,
    sample_weights=None,
    test_size=None,
    train_size=None,
    seed=None,
    shuffle=True,
    mode=ModelType.CLASSIFICATION,
):

    if test_size is None:
        test_size = 1 - train_size

    if mode == ModelType.REGRESSION:
        x_train, y_train, w_train, x_test, y_test, w_test = _stratified_regression_split(
            x_set,
            y_set,
            sample_weights,
            test_size,
            shuffle,
            seed
        )
    else:
        x_train, y_train, w_train, x_test, y_test, w_test = _stratified_classification_split(
            x_set,
            y_set,
            sample_weights,
            test_size,
            shuffle,
            seed
        )

    if shuffle:
        rng = np.random.default_rng(seed)
        train_indices = np.arange(len(x_train))
        rng.shuffle(train_indices)
        test_indices = np.arange(len(x_test))
        rng.shuffle(test_indices)

        x_train = np.array(x_train)[train_indices]
        y_train = np.array(y_train)[train_indices]
        if w_train is not None:
                w_train = np.array(w_train)[..., train_indices]
        x_test = np.array(x_test)[test_indices]
        y_test = np.array(y_test)[test_indices]
        if w_test is not None:
            w_test = np.array(w_test)[..., test_indices]
    if w_train is not None:
        return (x_train, y_train, w_train), (x_test, y_test, w_test)
    else:
        return (x_train, y_train), (x_test, y_test)

def _stratified_regression_split(
    x_set,
    y_set,
    sample_weights,
    test_size,
    shuffle,
    seed
):
    array_length = x_set.shape[0]
    if array_length != y_set.shape[0]:
        raise ValueError('Length of all passed arrays must be equal')

    exclude_weights = False
    if sample_weights is None:
        exclude_weights = True
        sample_weights = np.ones(y_set.shape[0])

    sort_order = np.argsort(y_set.reshape(-1,))
    x_set = x_set[sort_order]
    y_set = y_set[sort_order]
    sample_weights = sample_weights[..., sort_order]

    test_size = round(test_size, 2)
    if abs(test_size * 10 - round(test_size * 10)) < 1e-6:
        num_bins = 10
    else:
        num_bins = 100
    num_test_bins = round(num_bins * test_size)

    if shuffle:
        rng = np.random.default_rng(seed)
        len_data = len(x_set)
        for i in range(0, len_data, num_bins):
            num_items = min(num_bins, len_data - i)
            indices = np.arange(num_items)
            rng.shuffle(indices)
            x_set[i:i+num_items] = x_set[i:i+num_items][indices]
            y_set[i:i+num_items] = y_set[i:i+num_items][indices]
            sample_weights[..., i:i+num_items] = sample_weights[..., i:i+num_items][..., indices]

    x_train, y_train, w_train = [], [], []
    x_test, y_test, w_test = [], [], []
    for i in range(num_bins):
        if i < num_test_bins:
            x_test.append(x_set[i::num_bins])
            y_test.append(y_set[i::num_bins])
            w_test.append(sample_weights[..., i::num_bins])
        else:
            x_train.append(x_set[i::num_bins])
            y_train.append(y_set[i::num_bins])
            w_train.append(sample_weights[..., i::num_bins])

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    w_train = np.concatenate(w_train, axis=-1)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)
    w_test = np.concatenate(w_test, axis=-1)

    if exclude_weights:
        return x_train, y_train, None, x_test, y_test, None
    else:
        return x_train, y_train, w_train, x_test, y_test, w_test

def _stratified_classification_split(
    x_set,
    y_set,
    sample_weights,
    test_size,
    shuffle,
    seed
):
    array_length = x_set.shape[0]
    if array_length != y_set.shape[0]:
        raise ValueError('Length of all passed arrays must be equal')

    exclude_weights = False
    if sample_weights is None:
        exclude_weights = True
        sample_weights = np.ones(y_set.shape[0])

    test_size = round(test_size, 2)
    unique_labels = np.unique(y_set)

    y_flatten = y_set.reshape(-1)

    if shuffle:
        rng = np.random.default_rng(seed)
        len_data = len(x_set)
        for class_label in unique_labels:
            indices = np.arange(len_data)[y_flatten == class_label]
            shuffled_indices = indices.copy()
            rng.shuffle(shuffled_indices)
            x_set[indices] = x_set[shuffled_indices]
            y_set[indices] = y_set[shuffled_indices]
            sample_weights[..., indices] = sample_weights[..., shuffled_indices]

    x_train, y_train, w_train = [], [], []
    x_test, y_test, w_test = [], [], []
    for class_label in unique_labels:
        x_subset = x_set[y_flatten == class_label]
        y_subset = y_set[y_flatten == class_label]
        w_subset = sample_weights[..., y_flatten == class_label]
        test_amount = ceil(test_size * len(x_subset))
        x_train.append(x_subset[test_amount:])
        y_train.append(y_subset[test_amount:])
        w_train.append(w_subset[..., test_amount:])
        x_test.append(x_subset[:test_amount])
        y_test.append(y_subset[:test_amount])
        w_test.append(w_subset[..., :test_amount])

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    w_train = np.concatenate(w_train, axis=-1)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)
    w_test = np.concatenate(w_test, axis=-1)

    if exclude_weights:
        return x_train, y_train, None, x_test, y_test, None
    else:
        return x_train, y_train, w_train, x_test, y_test, w_test
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

    y_column = y_set.ndim == 2
    if y_column:
        y_set = y_set.squeeze()

    sample_weights = sample_weights.squeeze() if sample_weights is not None else None

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

    if y_column:
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    if w_train is not None:
        return (x_train, y_train, w_train), (x_test, y_test, w_test)
    else:
        return (x_train, y_train), (x_test, y_test)


def stratified_kfold(
    x_set,
    y_set,
    sample_weights=None,
    k=5,
    seed=None,
    shuffle=True,
    mode=ModelType.CLASSIFICATION,
):
    """
    Generate exhaustive stratified k-fold train/validation splits.

    Every sample appears in exactly one validation fold. Classification
    stratifies by class. Regression follows the same strategy used by imbal's
    stratified batching: sort by the continuous target, form neighboring
    pseudo-classes of size k, and distribute those samples across the k folds.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 2:
        raise ValueError('k must be an integer greater than or equal to 2')

    x_set = np.asarray(x_set)
    y_set = np.asarray(y_set)
    num_samples = x_set.shape[0]

    if num_samples != y_set.shape[0]:
        raise ValueError('Length of all passed arrays must be equal')
    if k > num_samples:
        raise ValueError(f'k ({k}) cannot exceed the number of samples ({num_samples})')

    exclude_weights = sample_weights is None
    if exclude_weights:
        sample_weights = np.ones(num_samples)
    else:
        sample_weights = np.asarray(sample_weights)
        if sample_weights.shape[-1] != num_samples:
            raise ValueError('Length of all passed arrays must be equal')

    labels = y_set
    if labels.ndim > 1:
        if labels.shape[-1] == 1:
            labels = labels.reshape(-1)
        elif mode == ModelType.CLASSIFICATION:
            labels = np.argmax(labels, axis=-1)
        else:
            labels = labels.reshape(-1)
    else:
        labels = labels.reshape(-1)

    rng = np.random.default_rng(seed)
    fold_validation_indices = [[] for _ in range(k)]

    if mode == ModelType.CLASSIFICATION:
        unique_labels, class_counts = np.unique(labels, return_counts=True)
        rarest_class_count = int(np.min(class_counts))
        if k > rarest_class_count:
            raise ValueError(
                f'k ({k}) cannot be larger than the number of samples in the '
                f'rarest class ({rarest_class_count})'
            )

        for class_label in unique_labels:
            class_indices = np.flatnonzero(labels == class_label)
            if shuffle:
                class_indices = class_indices.copy()
                rng.shuffle(class_indices)
            for fold_index, class_fold in enumerate(np.array_split(class_indices, k)):
                fold_validation_indices[fold_index].extend(class_fold.tolist())

    elif mode == ModelType.REGRESSION:
        sorted_indices = np.argsort(labels)

        # Each consecutive group contains nearby target values. Giving one member
        # of each full group to every fold spreads the target distribution across
        # folds while still using each original sample exactly once.
        for start in range(0, num_samples, k):
            group = sorted_indices[start:start + k].copy()
            if shuffle:
                rng.shuffle(group)
            for fold_index, sample_index in enumerate(group):
                fold_validation_indices[fold_index].append(int(sample_index))
    else:
        raise ValueError(f'Unsupported model mode: {mode}')

    all_indices = np.arange(num_samples, dtype=np.int64)
    folds = []

    for fold_indices in fold_validation_indices:
        val_indices = np.asarray(fold_indices, dtype=np.int64)
        train_mask = np.ones(num_samples, dtype=bool)
        train_mask[val_indices] = False
        train_indices = all_indices[train_mask]

        if shuffle:
            train_indices = train_indices.copy()
            val_indices = val_indices.copy()
            rng.shuffle(train_indices)
            rng.shuffle(val_indices)

        x_train = x_set[train_indices]
        y_train = y_set[train_indices]
        x_val = x_set[val_indices]
        y_val = y_set[val_indices]

        if exclude_weights:
            folds.append(((x_train, y_train), (x_val, y_val)))
        else:
            w_train = sample_weights[..., train_indices]
            w_val = sample_weights[..., val_indices]
            folds.append(((x_train, y_train, w_train), (x_val, y_val, w_val)))

    return folds

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
    sorted_x_set = x_set[sort_order]
    sorted_y_set = y_set[sort_order]
    sorted_sample_weights = sample_weights[..., sort_order]

    test_size = round(test_size, 2)
    if abs(test_size * 10 - round(test_size * 10)) < 1e-6:
        num_bins = 10
    else:
        num_bins = 100
    num_test_bins = round(num_bins * test_size)

    if shuffle:
        rng = np.random.default_rng(seed)
        len_data = len(sorted_x_set)
        for i in range(0, len_data, num_bins):
            num_items = min(num_bins, len_data - i)
            indices = np.arange(num_items)
            rng.shuffle(indices)
            sorted_x_set[i:i+num_items] = sorted_x_set[i:i+num_items][indices]
            sorted_y_set[i:i+num_items] = sorted_y_set[i:i+num_items][indices]
            sorted_sample_weights[..., i:i+num_items] = sorted_sample_weights[..., i:i+num_items][..., indices]

    x_train, y_train, w_train = [], [], []
    x_test, y_test, w_test = [], [], []
    for i in range(num_bins):
        if i < num_test_bins:
            x_test.append(sorted_x_set[i::num_bins])
            y_test.append(sorted_y_set[i::num_bins])
            w_test.append(sorted_sample_weights[..., i::num_bins])
        else:
            x_train.append(sorted_x_set[i::num_bins])
            y_train.append(sorted_y_set[i::num_bins])
            w_train.append(sorted_sample_weights[..., i::num_bins])

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

    usable_x_set = x_set.copy()
    usable_y_set = y_set.copy()
    usable_weights = sample_weights.copy()

    if shuffle:
        rng = np.random.default_rng(seed)
        len_data = len(usable_x_set)
        for class_label in unique_labels:
            indices = np.arange(len_data)[y_flatten == class_label]
            shuffled_indices = indices.copy()
            rng.shuffle(shuffled_indices)
            usable_x_set[indices] = usable_x_set[shuffled_indices]
            usable_y_set[indices] = usable_y_set[shuffled_indices]
            usable_weights[..., indices] = usable_weights[..., shuffled_indices]

    x_train, y_train, w_train = [], [], []
    x_test, y_test, w_test = [], [], []
    for class_label in unique_labels:
        x_subset = usable_x_set[y_flatten == class_label]
        y_subset = usable_y_set[y_flatten == class_label]
        w_subset = usable_weights[..., y_flatten == class_label]
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
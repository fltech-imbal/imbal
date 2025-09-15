from sklearn.model_selection import train_test_split
import numpy as np

def stratified_split(
        x_set,
        y_set,
        sample_weights=None,
        test_size=None,
        train_size=None,
        seed=0,
        shuffle=True,
        mode='class',
    ) -> tuple:

    if mode == 'reg':
        x_train, y_train, w_train, x_test, y_test, w_test = _stratified_regression_split(
            x_set,
            y_set,
            sample_weights=sample_weights,
            test_size=test_size,
            train_size=train_size,
            seed=seed,
            shuffle=shuffle,
        )
        if shuffle:
            rng = np.random.default_rng(seed)
            train_indices = np.arange(len(x_train))
            rng.shuffle(train_indices)
            test_indices = np.arange(len(x_test))
            rng.shuffle(test_indices)

            x_train = np.array(x_train)[train_indices]
            y_train = np.array(y_train)[train_indices]
            w_train = np.array(w_train)[train_indices]
            x_test = np.array(x_test)[test_indices]
            y_test = np.array(y_test)[test_indices]
            w_test = np.array(w_test)[test_indices]
    else:
        if shuffle:
            rng = np.random.default_rng(seed)
            indices = np.arange(x_set.shape[0])
            rng.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]
            sample_weights = sample_weights[indices]
        x_train, y_train, w_train, x_test, y_test, w_test = train_test_split(
            x_set,
            y_set,
            sample_weights,
            test_size=test_size,
            train_size=train_size,
            random_state=seed,
            stratify=y_set
        )


    return (x_train, y_train, w_train), (x_test, y_test, w_test)

def _stratified_regression_split(
        x_set,
        y_set,
        sample_weights,
        test_size=None,
        train_size=None,
        seed=None,
        shuffle=True
) -> list:
    if train_size is None:
        train_size = 1 - test_size

    array_length = x_set.shape[0]
    if array_length != y_set.shape[0]:
        raise ValueError('Length of all passed arrays must be equal')

    train_size = round(train_size, 2)
    train_per_batch = round(100*train_size)
    if abs(train_size * 10 - round(train_size * 10)) < 1e-6:
        batch_size = 10
        train_per_batch = round(10*train_size)
    else:
        batch_size = 100


    train_split_arrays = []
    test_split_arrays = []

    combined_arrays = np.array([x_set, y_set, sample_weights])

    for i in range(array_length // batch_size):
        batch = combined_arrays[:, i*batch_size:(i+1)*batch_size]
        indices = np.arange(batch_size)
        np.random.shuffle(indices)
        batch = batch[:, indices]
        train_split_arrays.append(batch[:, :train_per_batch])
        test_split_arrays.append(batch[:, train_per_batch:])

    if array_length / batch_size != round(array_length / batch_size):
        batch = combined_arrays[:, array_length // batch_size * batch_size:]
        partial_batch_size = len(batch[0])
        partial_train = round(partial_batch_size * train_size)
        indices = np.arange(partial_batch_size)
        np.random.shuffle(indices)
        batch = batch[:, indices]
        train_split_arrays.append(batch[:, :partial_train])
        if partial_train != partial_batch_size:
            test_split_arrays.append(batch[:, partial_train:])

    train_split_arrays = np.concatenate(train_split_arrays, axis=1)
    indices = np.arange(train_split_arrays.shape[1])
    np.random.shuffle(indices)
    train_split_arrays = train_split_arrays[:, indices]

    test_split_arrays = np.concatenate(test_split_arrays, axis=1)
    indices = np.arange(test_split_arrays.shape[1])
    np.random.shuffle(indices)
    test_split_arrays = test_split_arrays[:, indices]

    return train_split_arrays.tolist() + test_split_arrays.tolist()
        

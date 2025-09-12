from sklearn.model_selection import train_test_split
import numpy as np

def stratified_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True) -> list:
    return train_test_split(
        *arrays,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=arrays[1]
    )

def stratified_regression_split(
        *arrays,
        test_size=None,
        train_size=None,
        random_state=None,
        shuffle=True
) -> list:
    if train_size is None:
        train_size = 1 - test_size

    array_length = len(arrays[0])
    for i in range(len(arrays)):
        if array_length != len(arrays[i]):
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

    combined_arrays = np.array(arrays)

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
        

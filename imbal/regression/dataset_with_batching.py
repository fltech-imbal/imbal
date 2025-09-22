from imbal.util.stratified_sampling.dataset_with_batching import DatasetWithBatching as UtilDataset
from imbal.util.constants import ModelType

class DatasetWithBatching(UtilDataset):
    def __init__(self,
                 x_set,
                 y_set,
                 sample_weights=None,
                 batch_size=64,
                 num_batches=None,
                 seed=0,
                 shuffle=True,
                 sort='descending',
                 **kwargs
                 ) -> None:
        super(DatasetWithBatching, self).__init__(
            x_set,
            y_set,
            sample_weights=sample_weights,
            batch_size=batch_size,
            num_batches=num_batches,
            seed=seed,
            shuffle=shuffle,
            mode=ModelType.REGRESSION,
            sort=sort,
            **kwargs
        )
from imbal.util.stratified_sampling.split import split as util_split
from imbal.util.constants import ModelType

def split(
        x_set,
        y_set,
        sample_weights=None,
        test_size=None,
        train_size=None,
        seed=None,
        shuffle=True
    ) -> tuple:
    return util_split(
        x_set,
        y_set,
        sample_weights=sample_weights,
        test_size=test_size,
        train_size=train_size,
        seed=seed,
        shuffle=shuffle,
        mode=ModelType.REGRESSION
    )
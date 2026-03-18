import numpy as np
import imbal.util.backend as backend
from keras.src.trainers.data_adapters import data_adapter_utils
from imbal.util.backend.tools import verify_weight_scale


def decoupled_fit_with_cross_val_stage_one(
        model,
        x=None,
        y=None,
        sample_weight=None,
        validation_data=None,
        validation_split=None,
        epochs=1,
        batch_size=32,
        shuffle=True,
        stratify_batches=False,
        **kwargs
):
    """
    TODO: decoupled fit description

    Args:
        x:
        y:
        sample_weight:
        validation_data:
        validation_split:
        epochs:
        **kwargs:

    Returns:

    """
    if not model._mode_enum or not model._mode_subpackage:
        raise NotImplementedError

    training_model = model

    if validation_split and validation_data is None:
        (x, y, sample_weight), (val_x, val_y, val_sample_weight) = model._mode_subpackage.split(
            x,
            y,
            sample_weights=sample_weight,
            test_size=validation_split,
        )
        val_sample_weight = verify_weight_scale(val_sample_weight, show_warning=False)
        validation_data = (val_x, val_y, val_sample_weight)

    stage_one_x = x
    stage_one_y = y
    if model._use_decoder_branch:
        training_model = model._extended_model
        stage_one_y = [y, x]

    if validation_data is not None:
        if isinstance(validation_data, model._mode_subpackage.DatasetWithBatching):
            val_x, val_y, val_sample_weight = validation_data.unpack()
        else:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)
        if model._use_decoder_branch:
            val_y = [val_y, val_x]
        val_sample_weight = np.ones(val_sample_weight.shape)
        validation_data = (val_x, val_y, val_sample_weight)

    if model._use_decoder_branch and "validation_data" in kwargs and kwargs["validation_data"] is not None:
        val_x, val_y, val_sw = data_adapter_utils.unpack_x_y_sample_weight(kwargs["validation_data"])
        # match 2-output model: [class_target, recon_target]
        val_y = [val_y, val_x]
        kwargs["validation_data"] = (val_x, val_y, val_sw)

    stage_one_sample_weights = np.ones(x.shape[0])

    if stratify_batches:
        if model._use_decoder_branch:
            stage_one_x = backend.MultiDatasetWithBatching(
                x,
                stage_one_y,
                sample_weights=stage_one_sample_weights,
                batch_size=batch_size,
                shuffle=shuffle,
                multi_output=True,
                output_label_index=0,
                mode=model._mode_enum
            )
        else:
            stage_one_x = model._mode_subpackage.DatasetWithBatching(
                x,
                stage_one_y,
                sample_weights=stage_one_sample_weights,
                batch_size=batch_size,
                shuffle=shuffle
            )
        stage_one_y = None
        stage_one_sample_weights = None

    stage_one_history = training_model.fit(
        x=stage_one_x,
        y=stage_one_y,
        sample_weight=stage_one_sample_weights,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=epochs,
        **kwargs
    )

    return stage_one_history  # In the future, potentially only second stage history is returned

def decoupled_fit_with_cross_val_stage_two(
        model,
        x=None,
        y=None,
        epochs=1,
        sample_weight=None,
        batch_size=32,
        shuffle=True,
        stratify_batches=False,
        **kwargs
):

    model._use_decoder_branch = False

    stage_two_history = model.balanced_fit(
        x=x,
        y=y,
        epochs=epochs,
        sample_weight=sample_weight,
        batch_size=batch_size,
        shuffle=shuffle,
        stratify_batches=stratify_batches,
        **kwargs
    )

    return stage_two_history
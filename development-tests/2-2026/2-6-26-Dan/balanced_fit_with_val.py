import tensorflow as tf
import numpy as np
import keras, warnings, math
from keras.src.saving import serialization_lib
from keras.src.trainers.data_adapters import data_adapter_utils

import imbal
import imbal.util.backend as backend
from imbal.util.backend.tools import verify_weight_scale
from cross_validation import fit_k_folds
from imbal.util.backend.constants import ModelType


def fit_with_val(
        model,
        x=None,
        y=None,
        sample_weight=None,
        validation_data=None,
        validation_split=None,
        batch_size=32,
        shuffle=True,
        stratify_batches=False,
        do_cross_validation=False,
        num_folds=5,
        mode=ModelType.CLASSIFICATION,
        seed=0,
        **kwargs
):
    """
    TODO: Fit function description

    Args:
        x:
        y:
        sample_weight:
        validation_data:
        validation_split:
        batch_size:
        shuffle:
        do_cross_validation:
        **kwargs:

    Returns:

    """
    if not model._mode_enum or not model._mode_subpackage:
        raise NotImplementedError

    if stratify_batches or model._use_decoder_branch:
        x, y, sample_weight, stratify_batches = model._x_y_weight_split_data(x, y, sample_weight, stratify_batches)

    if sample_weight is None and not isinstance(x, tf.data.Dataset) and not isinstance(x, keras.utils.PyDataset):
        sample_weight = np.ones(x.shape[0])

    sample_weight = verify_weight_scale(sample_weight)

    if validation_split and validation_data is None:
        (x, y, sample_weight), (val_x, val_y, val_sample_weight) = model._mode_subpackage.split(
            x,
            y,
            sample_weights=sample_weight,
            test_size=validation_split,
        )
        sample_weight = verify_weight_scale(sample_weight, show_warning=False)
        val_sample_weight = verify_weight_scale(val_sample_weight, show_warning=False)
        validation_data = (val_x, val_y, val_sample_weight)

    training_model = model
    if model._use_decoder_branch:
        training_model = model._extended_model
        y = [y, x]

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
        val_sample_weight = verify_weight_scale(val_sample_weight)
        validation_data = (val_x, val_y, val_sample_weight)

    if stratify_batches:
        if model._use_decoder_branch:
            x = backend.MultiDatasetWithBatching(
                x,
                y,
                sample_weights=sample_weight,
                batch_size=batch_size,
                shuffle=shuffle,
                multi_output=True,
                output_label_index=0,
                mode=model._mode_enum
            )
        else:
            print("BEFORE WRAP:", y.shape, "pos", int(y.sum()), "neg", int((1 - y).sum()))
            x = model._mode_subpackage.DatasetWithBatching(
                x,
                y,
                sample_weights=sample_weight,
                batch_size=batch_size,
                shuffle=shuffle
            )
        y = None
        sample_weight = None

    if do_cross_validation:
        history = fit_k_folds(
            model=model, x=x, y=y,
            batch_size=batch_size,
            shuffle=shuffle,
            sample_weight=sample_weight,
            num_folds=num_folds,
            mode=mode,
            seed=seed,
            **kwargs,
        )
    else:
        history = keras.Model.fit(
            training_model,
            x=x,
            y=y,
            sample_weight=sample_weight,
            validation_split=validation_split,
            validation_data=validation_data,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )

    model._use_decoder_branch = model._generate_decoder_branch

    return history


def balanced_fit_with_val(
        model,
        x=None,
        y=None,
        class_weight=None,
        sample_density=None,
        sample_weight=None,
        validation_data=None,
        validation_split=None,
        batch_size=32,
        shuffle=True,
        stratify_batches=False,
        do_cross_validation=False,
        seed=0,
        num_folds=5,
        **kwargs
):
    """
    TODO: balanced fit description

    Args:
        x:
        y:
        class_weight:
        sample_density:
        sample_weight:
        validation_data:
        validation_split:
        batch_size:
        shuffle:
        **kwargs:

    Returns:

    """
    if not model._mode_enum or not model._mode_subpackage:
        raise NotImplementedError

    if stratify_batches or model._use_decoder_branch:
        x, y, sample_weight, stratify_batches = model._x_y_weight_split_data(x, y, sample_weight, stratify_batches)

    if sample_weight is None and not isinstance(x, tf.data.Dataset) and not isinstance(x, keras.utils.PyDataset):
        sample_weight = model._auto_compute_weights(y, sample_weight, class_weight, sample_density)

    if validation_data is not None:
        if isinstance(validation_data, model._mode_subpackage.DatasetWithBatching):
            val_x, val_y, val_sample_weight = validation_data.unpack()
        else:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)
        if val_sample_weight is None:
            combined_y = np.concatenate((y, val_y), axis=0)
            combined_weights = model._auto_compute_weights(combined_y, None, class_weight, None)
            sample_weight = combined_weights[:y.shape[0]]
            sample_weight = verify_weight_scale(sample_weight, show_warning=False)
            val_sample_weight = combined_weights[y.shape[0]:]
            val_sample_weight = verify_weight_scale(val_sample_weight, show_warning=False)
            validation_data = (val_x, val_y, val_sample_weight)

    if validation_split and validation_data is None:
        (x, y, sample_weight), (val_x, val_y, val_sample_weight) = model._mode_subpackage.split(
            x,
            y,
            sample_weights=sample_weight,
            test_size=validation_split,
        )
        sample_weight = verify_weight_scale(sample_weight, show_warning=False)
        val_sample_weight = verify_weight_scale(val_sample_weight, show_warning=False)
        validation_data = (val_x, val_y, val_sample_weight)

    return fit_with_val(
        model=model,
        x=x,
        y=y,
        sample_weight=sample_weight,
        batch_size=batch_size,
        shuffle=shuffle,
        stratify_batches=stratify_batches,
        validation_data=validation_data,
        validation_split=validation_split,
        do_cross_validation=do_cross_validation,
        num_folds=num_folds,
        seed=seed,
        **kwargs
    )
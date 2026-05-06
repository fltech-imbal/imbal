import tensorflow as tf
import numpy as np
import keras, warnings
from keras.src.trainers.data_adapters import data_adapter_utils
import imbal
import imbal.util.backend as backend
from imbal.util.backend.tools import verify_weight_scale
from imbal.util.backend.constants import ModelType
from modular_cross_validation import (
    fit_k_folds_modular,
    BalancedFitStrategy,
    RegularFitStrategy,
)

def decoupled_fit_with_cross_val(
    model,
    x=None,
    y=None,
    batch_size=32,
    shuffle=True,
    num_folds=5,
    seed=0,
    stage_one_params=None,
    stage_two_params=None,
    mode=ModelType.CLASSIFICATION,
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

    # if isinstance(epochs, tuple):
    #     first_train_epochs, second_train_epochs = epochs
    # else:
    #     first_train_epochs = epochs
    #     second_train_epochs = None

    # if validation_split and validation_data is None:
    #     (x, y, sample_weight), (val_x, val_y, val_sample_weight) = model._mode_subpackage.split(
    #         x,
    #         y,
    #         sample_weights=sample_weight,
    #         test_size=validation_split,
    #     )
    #     sample_weight = verify_weight_scale(sample_weight, show_warning=False)
    #     val_sample_weight = verify_weight_scale(val_sample_weight, show_warning=False)
    #     validation_data = (val_x, val_y, val_sample_weight)

    stage_one_x = x
    stage_one_y = y
    if model._use_decoder_branch:
       training_model = model._extended_model
       stage_one_y = [y, x]

    # val_x, stage_two_val_y, stage_two_val_sample_weight = None, None, None
    # if validation_data is not None:
    #     if isinstance(validation_data, model._mode_subpackage.DatasetWithBatching):
    #         val_x, val_y, val_sample_weight = validation_data.unpack()
    #     else:
    #         (
    #             val_x,
    #             val_y,
    #             val_sample_weight,
    #         ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)
    #         stage_two_val_y = val_y
    #     if model._use_decoder_branch:
    #         val_y = [val_y, val_x]
    #     stage_two_val_sample_weight = verify_weight_scale(val_sample_weight)
    #     val_sample_weight = np.ones(val_sample_weight.shape)
    #     validation_data = (val_x, val_y, val_sample_weight)

    stage_one_sample_weights = np.ones(x.shape[0])

    # if stratify_batches:
    #     if model._use_decoder_branch:
    #         stage_one_x = backend.MultiDatasetWithBatching(
    #             x,
    #             stage_one_y,
    #             sample_weights=stage_one_sample_weights,
    #             batch_size=batch_size,
    #             shuffle=shuffle,
    #             multi_output=True,
    #             output_label_index=0,
    #             mode=model._mode_enum
    #         )
    #     else:
    #         stage_one_x = model._mode_subpackage.DatasetWithBatching(
    #             x,
    #             stage_one_y,
    #             sample_weights=stage_one_sample_weights,
    #             batch_size=batch_size,
    #             shuffle=shuffle
    #         )
    #     stage_one_y = None
    #     stage_one_sample_weights = None

    stage_one_params.x = stage_one_x
    stage_one_params.y = stage_one_y
    stage_one_params.sample_weight = stage_one_sample_weights

    stage_one_history = fit_k_folds_modular(
        training_model,
        stage_one_x,
        stage_one_y,
        strategy=RegularFitStrategy(),
        params=stage_one_params,
        batch_size=batch_size,
        num_folds=num_folds,
        shuffle=shuffle,
        seed=seed,
        mode=mode,
    )

    # stage_one_history = training_model.fit(
    #     x=stage_one_x,
    #     y=stage_one_y,
    #     sample_weight=stage_one_sample_weights,
    #     validation_data=validation_data,
    #     validation_split=validation_split,
    #     epochs=first_train_epochs,
    #     **kwargs
    # )

    representation_layer_index = backend.tools.positive_model_layer_index(model, model._representation_layer_index)
    found_layer, found_index = imbal.util.get_representation_layer_index(
        model,
        desired_layer_index=representation_layer_index
    )
    if found_index is None:
        raise ValueError(
            "Unable to find viable representation layer. Please ensure you model has at least two trainable layers")
    if representation_layer_index > found_index:
        warnings.warn(
            f"Overriding representation layer to layer {found_index} (originally {representation_layer_index})")
        representation_layer_index = found_index

    untrainable_layers = model.layers[:representation_layer_index + 1]
    trainable_layers = model.layers[representation_layer_index + 1:]

    for layer in trainable_layers:
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            layer.set_weights([layer.kernel_initializer(shape=np.asarray(layer.kernel.shape)),
                               layer.bias_initializer(shape=np.asarray(layer.bias.shape))])
    for layer in untrainable_layers:
        layer.trainable = False
    if model._use_decoder_branch:
        for layer in model._decoder_branch:
            layer.trainable = False

    # second_stage_fit_kwargs = kwargs.copy()
    # second_stage_fit_kwargs['epochs'] = len(
    #     stage_one_history.epoch) if second_train_epochs is None else second_train_epochs
    # second_stage_fit_kwargs['sample_weight'] = sample_weight
    # second_stage_fit_kwargs['validation_data'] = None if validation_data is None else (val_x, stage_two_val_y,
    #                                                                                    stage_two_val_sample_weight)
    # second_stage_fit_kwargs['validation_split'] = validation_split
    # second_stage_fit_kwargs['callbacks'] = None
    # second_stage_fit_kwargs['shuffle'] = shuffle
    # second_stage_fit_kwargs['batch_size'] = batch_size
    # second_stage_fit_kwargs['stratify_batches'] = stratify_batches
    # second_stage_fit_kwargs.update(model._second_stage_fit_kwargs)

    model._use_decoder_branch = False

    stage_two_params.stratify_batches = False

    stage_two_history = fit_k_folds_modular(
        model,
        x,
        y,
        strategy=BalancedFitStrategy(),
        params=stage_two_params,
        batch_size=batch_size,
        num_folds=num_folds,
        shuffle=shuffle,
        seed=seed,
        mode=mode,
    )

    model._use_decoder_branch = model._generate_decoder_branch

    if model._generate_decoder_branch:
        model._extended_model.trainable = True

    return stage_one_history, stage_two_history  # In the future, potentially only second stage history is returned

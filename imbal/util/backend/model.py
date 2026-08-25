import tensorflow as tf
import numpy as np
import keras, warnings, copy, gc
from keras.src.saving import serialization_lib

import imbal
import imbal.util.backend as backend
from imbal.util.backend.constants import ModelType
from imbal.util.backend.tools import verify_weight_scale

@keras.utils.register_keras_serializable()
def mse_reconstruction_loss(y_true, y_pred):
    sq = tf.math.squared_difference(y_true, y_pred)
    axes = tf.range(1, tf.rank(sq))
    loss_per_example = tf.reduce_mean(sq, axis=axes)
    return loss_per_example

def _clone_callbacks(callbacks):
    if callbacks is None:
        return None
    cloned = []
    for cb in callbacks:
        if hasattr(cb, "get_config"):
            try:
                config = cb.get_config()
                cls = cb.__class__
                new_cb = cls.from_config(config)
                cloned.append(new_cb)
                continue
            except Exception:
                pass
        try:
            cloned.append(copy.deepcopy(cb))
            continue
        except Exception:
            pass
        raise RuntimeError("Unable to create copy of passed callbacks")

    return cloned

@tf.keras.utils.register_keras_serializable()
class Model(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.best_sample_weights = kwargs.get('best_sample_weights', None)
        self.best_class_weights = kwargs.get('best_class_weights', None)
        self.best_decision_threshold = kwargs.get('best_decision_threshold', None)
        self.best_weight_index = kwargs.get('best_weight_index', None)
        self._compare_metric_spec = kwargs.get('_compare_metric_spec', None)
        self._compare_metric_uses_weights = kwargs.get('_compare_metric_uses_weights', False)

        self._generate_decoder_branch = kwargs.get('_generate_decoder_branch', False)
        self._use_decoder_branch = kwargs.get('_use_decoder_branch', False)
        self._representation_layer_index = kwargs.get('_representation_layer_index', -2)
        self._extended_model = kwargs.get('_extended_model', None)
        self._decoder_branch = kwargs.get('_decoder_branch', None)
        self._second_stage_fit_kwargs = kwargs.get('_second_stage_fit_kwargs', {})
        self._mode_subpackage = kwargs.get('_mode_subpackage', None)
        self._mode_enum = kwargs.get('_mode_enum', None)
        self._compiled_optimizer = kwargs.get('_compiled_optimizer', None)
        self._reconstruction_lambda = kwargs.get('_reconstruction_lambda', None)

    def get_config(self):
        config = super().get_config()
        config.update({
            'best_sample_weights' : self.best_sample_weights,
            'best_class_weights' : self.best_class_weights,
            'best_decision_threshold' : self.best_decision_threshold,
            'best_weight_index' : self.best_weight_index,
            '_compare_metric_spec' : self._compare_metric_spec,
            '_compare_metric_uses_weights' : self._compare_metric_uses_weights,
            '_generate_decoder_branch' : self._generate_decoder_branch,
            '_use_decoder_branch' : self._use_decoder_branch,
            '_representation_layer_index' : self._representation_layer_index,
            '_extended_model' : self._extended_model,
            '_decoder_branch' : self._decoder_branch,
            '_second_stage_fit_kwargs' : self._second_stage_fit_kwargs,
            '_mode_subpackage' : self._mode_subpackage,
            '_mode_enum' : self._mode_enum.value if self._mode_enum is not None else None,
            '_compiled_optimizer' : self._compiled_optimizer,
            '_reconstruction_lambda' : self._reconstruction_lambda
        })
        return config

    def fit(
        self,
        x=None,
        y=None,
        sample_weight=None,
        candidate_evaluation_sample_weight=None,
        validation_data=None,
        validation_split=None,
        epochs=1,
        batch_size=32,
        shuffle=True,
        stratify_batches=True,
        verbose_imbal=1,
        seed=None,
        **kwargs
    ):
        """
        An extension of `TensorFlow's model.fit function <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_
        that allows for data batches to be stratified if desired.

        Args:
            x: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                A NumPy array of data points, arranged as a column vector.
            y: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                A NumPy array of labels, arranged as a row vector, column vector, or list of one-hot vectors.
            sample_weight: Optional, default :code:`None`. A list of sample weights.
                Optionally, a 2D list of sample weights can be provided, in which case
                the model will be fit once for each list of sample weights provided, with the final model weights being set to the
                final weights from the fit which best optimizes the first metric passed during :code:`Model.compile`.
                See "Using Multiple Weight Candidates" below for more details.
            candidate_evaluation_sample_weight: Optional, default :code:`None`.
                When performing a fit with multiple weights candidates, determines what sample weighting should be used when computing
                the metric to compare weight candidate performance (See "Using Multiple Weight Candidates below for more details).
            validation_data: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                The data used to validate the model during training.
                See `Tensorflow's model.fit documentation <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_.
            validation_split: Optional, default :code:`None`. A float value representing the proportion of the
                    provided training data to split off into a separate dataset used for model validation.
            epochs: Optional, default :code:`1`. (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                The number of epochs to train the model for.
            batch_size: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                The batch size to use during training.
            shuffle: Optional, default :code:`True` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                Whether to shuffle the data before each epoch.
            stratify_batches: Optional, default :code:`True`. Whether to stratify data batch-wise during training.
                See :doc:`DatasetWithBatching </imbal/classification/dataset_with_batching>` for details.
                Only used when :code:`multi_output` is :code:`True`.
            verbose_imbal: Optional, default :code:`1`. The verbosity level of debug messages associated with
                imbal functionality. When set to :code:`0`, no imbal debug messages will print. When set to :code:`1`,
                general debug messages will be printed. When greater than :code:`1`, all messages will be printed.
            **kwargs: Any additional keyword arguments accepted by `TensorFlow's model.fit function <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_

        Returns:
            A History object. Its History.history attribute is a record of training loss values and metrics values
            at successive epochs, as well as validation loss values and validation metrics values (if applicable).

        Example:

        .. code-block:: python

            # Assume data has been loaded as '(x_train, y_train), (x_test, y_test)', and
            # we have a compiled model

            model.fit(
                x_train,
                y_train,
                batch_size=64,
                validation_split=0.2
            )

        """

        if not self._mode_enum or not self._mode_subpackage:
            raise NotImplementedError

        return self._enforced_fit(
            x=x,
            y=y,
            class_weight=None,
            sample_density=None,
            sample_weight=sample_weight,
            candidate_evaluation_sample_weight=candidate_evaluation_sample_weight,
            validation_data=validation_data,
            validation_densities=None,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            stratify_batches=stratify_batches,
            verbose_imbal=verbose_imbal,
            require_weighting=False,
            seed=None,
            **kwargs
        )

    def _balanced_fit(
        self,
        x=None,
        y=None,
        class_weight=None,
        sample_density=None,
        sample_weight=None,
        candidate_evaluation_sample_weight=None,
        candidate_evaluation_class_weight=None,
        validation_data=None,
        validation_densities=None,
        validation_split=None,
        epochs=1,
        batch_size=32,
        shuffle=True,
        stratify_batches=True,
        verbose_imbal=1,
        seed=None,
        **kwargs
    ):
        if not self._mode_enum or not self._mode_subpackage:
            raise NotImplementedError

        return self._enforced_fit(
            x=x,
            y=y,
            class_weight=class_weight,
            sample_density=sample_density,
            sample_weight=sample_weight,
            candidate_evaluation_sample_weight=candidate_evaluation_sample_weight,
            candidate_evaluation_class_weight=candidate_evaluation_class_weight,
            validation_data=validation_data,
            validation_densities=validation_densities,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            stratify_batches=stratify_batches,
            verbose_imbal=verbose_imbal,
            seed=seed,
            require_weighting=True,
            **kwargs
        )

    def _decoupled_fit(
        self,
        x=None,
        y=None,
        class_weight=None,
        sample_weight=None,
        sample_density=None,
        candidate_evaluation_sample_weight=None,
        candidate_evaluation_class_weight=None,
        validation_data=None,
        validation_densities=None,
        validation_split=None,
        epochs=1,
        batch_size=32,
        shuffle=True,
        stratify_batches=True,
        verbose_imbal=1,
        seed=None,
        **kwargs
    ):

        if not self._mode_enum or not self._mode_subpackage:
            raise NotImplementedError

        if isinstance(epochs, tuple):
            stage_one_epochs, stage_two_epochs = epochs
        else:
            stage_one_epochs = epochs
            stage_two_epochs = None


        stage_one_sample_weights = np.ones(x.shape[0])

        if validation_data is not None:
            x_val, y_val, w_val = self._unpack_validation(validation_data)
            stage_one_validation = (x_val, y_val, np.ones(x_val.shape[0]))
        else:
            stage_one_validation = None

        first_stage_fit_kwargs = kwargs.copy()
        if 'callbacks' in kwargs:
            first_stage_fit_kwargs['callbacks'] = _clone_callbacks(kwargs['callbacks'])

        stage_one_history = self._enforced_fit(
            x=x,
            y=y,
            class_weight=None,
            sample_density=None,
            sample_weight=stage_one_sample_weights,
            candidate_evaluation_sample_weight=candidate_evaluation_sample_weight,
            candidate_evaluation_class_weight=candidate_evaluation_class_weight,
            validation_data=stage_one_validation,
            validation_densities=None,
            validation_split=validation_split,
            batch_size=batch_size,
            shuffle=shuffle,
            stratify_batches=stratify_batches,
            verbose_imbal=verbose_imbal,
            seed=seed,
            require_weighting=False,
            epochs=stage_one_epochs,
            **first_stage_fit_kwargs
        )

        representation_layer_index = backend.tools.positive_model_layer_index(self, self._representation_layer_index)
        found_layer, found_index = imbal.util.get_representation_layer_index(
            self,
            desired_layer_index=representation_layer_index
        )
        if found_index is None:
            raise ValueError(
                "Unable to find viable representation layer. Please ensure you model has at least two trainable layers")
        if representation_layer_index > found_index:
            warnings.warn(
                f"Overriding representation layer to layer {found_index} (originally {representation_layer_index})")
            representation_layer_index = found_index

        untrainable_layers = self.layers[:representation_layer_index + 1]
        trainable_layers = self.layers[representation_layer_index + 1:]
        for layer in trainable_layers:
            if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
                layer.set_weights([layer.kernel_initializer(shape=np.asarray(layer.kernel.shape)),
                                   layer.bias_initializer(shape=np.asarray(layer.bias.shape))])
        for layer in untrainable_layers:
            layer.trainable = False
        if self._use_decoder_branch:
            for layer in self._decoder_branch:
                layer.trainable = False

        second_stage_fit_kwargs = kwargs.copy()
        if 'callbacks' in kwargs:
            second_stage_fit_kwargs['callbacks'] = _clone_callbacks(kwargs['callbacks'])
        self._reset_optimizer(self)

        # Allow second stage overrides
        second_stage_fit_kwargs.update(self._second_stage_fit_kwargs)

        self._use_decoder_branch = False
        stage_two_history = self._enforced_fit(
            x=x,
            y=y,
            class_weight=class_weight,
            sample_density=sample_density,
            sample_weight=sample_weight,
            candidate_evaluation_sample_weight=candidate_evaluation_sample_weight,
            candidate_evaluation_class_weight=candidate_evaluation_class_weight,
            validation_data=validation_data,
            validation_densities=validation_densities,
            validation_split=validation_split,
            batch_size=batch_size,
            shuffle=shuffle,
            stratify_batches=stratify_batches,
            verbose_imbal=verbose_imbal,
            seed=seed,
            require_weighting=True,
            epochs=len(stage_one_history.epoch) if stage_two_epochs is None else stage_two_epochs,
            **second_stage_fit_kwargs
        )

        self._use_decoder_branch = self._generate_decoder_branch
        if self._generate_decoder_branch:
            self._extended_model.trainable = True

        return stage_one_history, stage_two_history  # In the future, potentially only second stage history is returned

    def _enforced_fit(
        self,
        x=None,
        y=None,
        class_weight=None,
        sample_density=None,
        sample_weight=None,
        candidate_evaluation_sample_weight=None,
        candidate_evaluation_class_weight=None,
        validation_data=None,
        validation_densities=None,
        validation_split=None,
        epochs=1,
        batch_size=32,
        shuffle=True,
        stratify_batches=True,
        verbose_imbal=1,
        seed=None,
        require_weighting=False,
        **kwargs
    ):
        self.best_sample_weights = None
        self.best_class_weights = None
        self.best_decision_threshold = None
        self.best_weight_index = None

        repeated_validation_split = validation_split is not None and validation_data is None

        (x, y, sample_weight), validation_data = self._prepare_training_data(
            x=x,
            y=y,
            class_weight=class_weight,
            sample_density=sample_density,
            sample_weight=sample_weight,
            validation_data=validation_data,
            validation_densities=validation_densities,
            validation_split=None if repeated_validation_split else validation_split,
            shuffle=shuffle,
            seed=seed,
            require_weighting=require_weighting
        )

        training_model = self._extended_model if self._use_decoder_branch else self
        initial_weights = training_model.get_weights()

        if repeated_validation_split:
            history = self._repeated_validation_split_fit(
                model=training_model,
                x=x,
                y=y,
                sample_weight=sample_weight,
                class_weight=class_weight,
                candidate_evaluation_sample_weight=candidate_evaluation_sample_weight,
                candidate_evaluation_class_weight=candidate_evaluation_class_weight,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                stratify_batches=stratify_batches,
                verbose_imbal=verbose_imbal,
                seed=seed,
                initial_weights=initial_weights,
                **kwargs
            )
            self._use_decoder_branch = self._generate_decoder_branch
            return history

        if sample_weight.shape[0] == 1:
            sample_weight = sample_weight.reshape(-1)

            if stratify_batches:
                x_train, y_train, w_train = self._stratify_data(x, y, sample_weight, batch_size, shuffle)
            else:
                x_train, y_train = x, y
                w_train = sample_weight

            if validation_data is not None:
                x_val, y_val, w_val = validation_data
                w_val = w_val.reshape(-1)
                validation_data = (x_val, y_val, w_val)

            if self._use_decoder_branch and self._reconstruction_lambda is None:
                if verbose_imbal > 0:
                    print(f'Determining reconstruction lambda...')
                self._reconstruction_lambda = self._determine_reconstruction_lambda(
                    training_model,
                    x=x_train,
                    y=y_train,
                    sample_weight=w_train,
                    validation_data=validation_data,
                    validation_split=validation_split,
                    batch_size=None if stratify_batches else batch_size,
                    shuffle=shuffle,
                    **kwargs
                )

                if verbose_imbal > 0:
                    print(f'Found reconstruction lambda {self._reconstruction_lambda:.4f}')

                config = training_model.get_compile_config()
                config['loss_weights'] = [1.0, float(self._reconstruction_lambda)]
                training_model.compile_from_config(config)

            history = keras.Model.fit(
                training_model,
                x=x_train,
                y=y_train,
                sample_weight=w_train,
                validation_data=validation_data,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=None if stratify_batches else batch_size,
                shuffle=shuffle,
                **kwargs
            )

            if class_weight is not None:
                self.best_class_weights = class_weight
            else:
                self.best_sample_weights = sample_weight

            if self._mode_enum == ModelType.CLASSIFICATION:
                _, best_threshold, _ = self._optimize_metric(
                    self,
                    x,
                    y,
                    sample_weight,
                    validation_data,
                    verbose_imbal
                )
                self.best_decision_threshold = best_threshold
        else:
            history = self._multi_weight_fit(
                model=training_model,
                x=x,
                y=y,
                sample_weight=sample_weight,
                class_weight=class_weight,
                candidate_evaluation_sample_weight=candidate_evaluation_sample_weight,
                candidate_evaluation_class_weight=candidate_evaluation_class_weight,
                validation_data=validation_data,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                stratify_batches=stratify_batches,
                verbose_imbal=verbose_imbal,
                **kwargs
            )

        if validation_data is not None:
            final_epochs = np.argmin(history.history['val_loss']) + 1
            x_val, y_val, w_val = validation_data
            x_final = np.concatenate((x, x_val), axis=0)
            if self._use_decoder_branch:
                y_final = [np.concatenate((y[i], y_val[i]), axis=0) for i in range(len(y))]
            else:
                y_final = np.concatenate((y, y_val), axis=0)

            if self.best_sample_weights is not None:
                w_final = np.concatenate((self.best_sample_weights, w_val if self.best_weight_index is None else w_val[self.best_weight_index]), axis=0)
            else:
                w_final = imbal.classification.generate_sample_weights(
                    y_final[0] if self._use_decoder_branch else y_final,
                    class_weight=self.best_class_weights
                )

            if verbose_imbal > 0:
                print('Performing final fit using combined training and validation data')

            if 'callbacks' in kwargs:
                kwargs['callbacks'] = None

            if stratify_batches:
                x_final, y_final, w_final = self._stratify_data(x_final, y_final, w_final, batch_size, shuffle)

            training_model.set_weights(initial_weights)
            self._reset_optimizer(training_model)
            history = keras.Model.fit(
                training_model,
                x=x_final,
                y=y_final,
                sample_weight=w_final,
                epochs=final_epochs,
                batch_size=None if stratify_batches else batch_size,
                shuffle=shuffle,
                **kwargs
            )

        self._use_decoder_branch = self._generate_decoder_branch

        return history

    def _repeated_validation_split_fit(
        self,
        model,
        x,
        y,
        sample_weight,
        class_weight,
        candidate_evaluation_sample_weight,
        candidate_evaluation_class_weight,
        validation_split,
        epochs,
        batch_size,
        shuffle,
        stratify_batches,
        verbose_imbal,
        seed,
        initial_weights,
        **kwargs
    ):
        num_validation_splits = 5
        primary_y = y[0] if self._use_decoder_branch else y
        sample_weight = np.asarray(sample_weight)

        validation_splits = []
        for split_index in range(num_validation_splits):
            split_seed = None if seed is None else seed + split_index
            (_, _, train_indices), (_, _, val_indices) = imbal.util.backend.split(
                x,
                primary_y,
                np.arange(x.shape[0]),
                test_size=validation_split,
                shuffle=shuffle,
                seed=split_seed,
                mode=self._mode_enum
            )
            validation_splits.append((
                np.asarray(train_indices, dtype=np.int64),
                np.asarray(val_indices, dtype=np.int64)
            ))

        if self._use_decoder_branch and self._reconstruction_lambda is None:
            if verbose_imbal > 0:
                print(f'Determining reconstruction lambda...')
            self._reconstruction_lambda = self._determine_reconstruction_lambda(
                model,
                x=x,
                y=y,
                validation_split=validation_split,
                batch_size=None if stratify_batches else batch_size,
                shuffle=shuffle,
                **kwargs
            )

            if verbose_imbal > 0:
                print(f'Found reconstruction lambda {self._reconstruction_lambda:.4f}')

            config = model.get_compile_config()
            config['loss_weights'] = [1.0, float(self._reconstruction_lambda)]
            model.compile_from_config(config)

        if sample_weight.shape[0] == 1:
            weights = sample_weight.reshape(-1)
            epoch_counts = []
            last_threshold = None

            for split_index, (train_indices, val_indices) in enumerate(validation_splits):
                model.set_weights(initial_weights)
                self._reset_optimizer(model)
                x_train = x[train_indices]
                x_val = x[val_indices]
                y_train_primary = primary_y[train_indices]
                y_val_primary = primary_y[val_indices]
                y_train = [y_train_primary, x_train] if self._use_decoder_branch else y_train_primary
                y_val = [y_val_primary, x_val] if self._use_decoder_branch else y_val_primary
                w_train = weights[train_indices]
                w_val = weights[val_indices]

                if stratify_batches:
                    fit_x, fit_y, fit_weights = self._stratify_data(
                        x_train, y_train, w_train, batch_size, shuffle
                    )
                else:
                    fit_x, fit_y, fit_weights = x_train, y_train, w_train

                current_kwargs = kwargs.copy()
                if 'callbacks' in kwargs:
                    current_kwargs['callbacks'] = _clone_callbacks(kwargs['callbacks'])

                history = keras.Model.fit(
                    model,
                    x=fit_x,
                    y=fit_y,
                    sample_weight=fit_weights,
                    validation_data=(x_val, y_val, w_val),
                    epochs=epochs,
                    batch_size=None if stratify_batches else batch_size,
                    shuffle=shuffle,
                    **current_kwargs
                )

                current_epochs = np.argmin(history.history['val_loss']) + 1
                epoch_counts.append(current_epochs)

                if self._mode_enum == ModelType.CLASSIFICATION:
                    _, last_threshold, _ = self._optimize_metric(
                        self,
                        x_train,
                        y_train,
                        w_val,
                        (x_val, y_val, w_val),
                        verbose_imbal
                    )

                if verbose_imbal > 0:
                    print(
                        f'Validation split {split_index + 1}/{num_validation_splits} '
                        f'epoch estimate: {current_epochs}'
                    )

            final_epochs = int(np.round(np.mean(epoch_counts)))
            if class_weight is not None:
                self.best_class_weights = class_weight
            else:
                self.best_sample_weights = weights
            self.best_decision_threshold = last_threshold

        else:
            weight_type = 'class weight' if class_weight is not None else 'sample weight'
            best_weights_index = None
            best_metric_result = None
            best_threshold = None
            best_average_epochs = None

            if candidate_evaluation_sample_weight is not None:
                candidate_evaluation_weights = np.asarray(candidate_evaluation_sample_weight)
            else:
                if self._mode_enum == ModelType.CLASSIFICATION:
                    candidate_evaluation_weights = imbal.classification.generate_sample_weights(
                        primary_y,
                        class_weight=candidate_evaluation_class_weight
                    )
                else:
                    raise RuntimeError(
                        'To perform a a fit with multiple weight candidates for regression, '
                        'some candidate evaluation weights must be specified'
                    )

            for index, weights in enumerate(sample_weight):
                candidate_epoch_counts = []
                candidate_metric_results = []
                candidate_thresholds = []
                compare_function = None

                for split_index, (train_indices, val_indices) in enumerate(validation_splits):
                    tf.keras.backend.clear_session()
                    model.set_weights(initial_weights)
                    self._reset_optimizer(model)

                    x_train = x[train_indices]
                    x_val = x[val_indices]
                    y_train_primary = primary_y[train_indices]
                    y_val_primary = primary_y[val_indices]
                    y_train = [y_train_primary, x_train] if self._use_decoder_branch else y_train_primary
                    y_val = [y_val_primary, x_val] if self._use_decoder_branch else y_val_primary
                    w_train = weights[train_indices]
                    w_val = weights[val_indices]

                    if stratify_batches:
                        fit_x, fit_y, fit_weights = self._stratify_data(
                            x_train, y_train, w_train, batch_size, shuffle
                        )
                    else:
                        fit_x, fit_y, fit_weights = x_train, y_train, w_train

                    current_kwargs = kwargs.copy()
                    if 'callbacks' in kwargs:
                        current_kwargs['callbacks'] = _clone_callbacks(kwargs['callbacks'])

                    if hasattr(self, 'compiled_metrics') and self.compiled_metrics:
                        reset_fn = getattr(self.compiled_metrics, 'reset_states', None) or getattr(
                            self.compiled_metrics, 'reset_state', None
                        )
                        if reset_fn:
                            reset_fn()

                    current_val_data = (x_val, y_val, w_val)
                    history = keras.Model.fit(
                        model,
                        x=fit_x,
                        y=fit_y,
                        sample_weight=fit_weights,
                        validation_data=current_val_data,
                        epochs=epochs,
                        batch_size=None if stratify_batches else batch_size,
                        shuffle=shuffle,
                        **current_kwargs
                    )

                    current_epochs = np.argmin(history.history['val_loss']) + 1
                    candidate_epoch_counts.append(current_epochs)

                    if candidate_evaluation_weights.ndim == 2:
                        evaluation_weights = candidate_evaluation_weights[index][val_indices]
                    else:
                        evaluation_weights = candidate_evaluation_weights[val_indices]

                    current_metric_result, current_threshold, compare_function = self._optimize_metric(
                        self,
                        x_train,
                        y_train,
                        evaluation_weights,
                        current_val_data,
                        verbose_imbal
                    )
                    candidate_metric_results.append(current_metric_result)
                    if current_threshold is not None:
                        candidate_thresholds.append(current_threshold)

                    if verbose_imbal > 0:
                        print(
                            f'[{index + 1}/{len(sample_weight)}] Validation split '
                            f'{split_index + 1}/{num_validation_splits} epoch estimate: {current_epochs}'
                        )

                average_epochs = int(np.round(np.mean(candidate_epoch_counts)))
                average_metric_result = np.mean(candidate_metric_results)
                average_threshold = (
                    float(np.mean(candidate_thresholds)) if len(candidate_thresholds) > 0 else None
                )

                if verbose_imbal > 0:
                    print(
                        f'[{index + 1}/{len(sample_weight)}] Average epoch estimate for '
                        f'{weight_type} candidate at index {index}: {average_epochs}'
                    )
                    print(
                        f'[{index + 1}/{len(sample_weight)}] Average comparison metric for '
                        f'{weight_type} candidate at index {index}: {average_metric_result:.4f}'
                    )

                if best_metric_result is None or compare_function(average_metric_result, best_metric_result):
                    best_metric_result = average_metric_result
                    best_weights_index = index
                    best_threshold = average_threshold
                    best_average_epochs = average_epochs

            if class_weight is not None:
                self.best_class_weights = class_weight[best_weights_index]
            else:
                self.best_sample_weights = sample_weight[best_weights_index]

            self.best_weight_index = best_weights_index
            self.best_decision_threshold = best_threshold
            final_epochs = best_average_epochs
            weights = sample_weight[best_weights_index]

            if verbose_imbal > 0:
                print(f'Best {weight_type} candidate index: {best_weights_index}')

        if verbose_imbal > 0:
            print(f'Average validation split epoch estimate: {final_epochs}')
            print('Performing final fit using all training data')

        final_kwargs = kwargs.copy()
        if 'callbacks' in final_kwargs:
            final_kwargs['callbacks'] = None

        if stratify_batches:
            x_final, y_final, w_final = self._stratify_data(
                x, y, weights, batch_size, shuffle
            )
        else:
            x_final, y_final, w_final = x, y, weights

        model.set_weights(initial_weights)
        self._reset_optimizer(model)
        history = keras.Model.fit(
            model,
            x=x_final,
            y=y_final,
            sample_weight=w_final,
            epochs=final_epochs,
            batch_size=None if stratify_batches else batch_size,
            shuffle=shuffle,
            **final_kwargs
        )

        gc.collect()
        tf.keras.backend.clear_session()

        return history

    def _reset_optimizer(self, model):
        model.optimizer = copy.deepcopy(self._compiled_optimizer)

    def _determine_reconstruction_lambda(
        self,
        model,
        x=None,
        y=None,
        validation_data=None,
        validation_split=None,
        batch_size=None,
        shuffle=True,
        **kwargs
    ):
        starting_model_weights = model.get_weights()

        current_kwargs = kwargs.copy()
        if 'callbacks' in kwargs:
            current_kwargs['callbacks'] = _clone_callbacks(kwargs['callbacks'])

        history = keras.Model.fit(
            model,
            x=x,
            y=y,
            validation_data=validation_data,
            validation_split=validation_split,
            epochs=100,
            batch_size=batch_size,
            shuffle=shuffle,
            **current_kwargs
        )

        combined_loss = np.array(history.history['loss'])
        half_length = len(combined_loss) // 2
        decoder_loss = np.array(history.history[[key for key in history.history if key.startswith('imbal')][0]])[half_length:]
        standard_loss = combined_loss[half_length:] - decoder_loss
        ratios = standard_loss / decoder_loss

        model.set_weights(starting_model_weights)
        self._reset_optimizer(model)

        return np.mean(ratios).astype(np.float32)


    def _optimize_metric(
        self,
        model,
        x,
        y,
        sample_weight,
        validation_data,
        verbose_imbal
    ):

        weights = None
        if self._compare_metric_spec is not None:
            compare_metric = self._retrieve_metric(self._compare_metric_spec)

            if self._compare_metric_uses_weights:
                weights = sample_weight
        else:
            if self._mode_enum == ModelType.CLASSIFICATION:
                compare_metric = keras.metrics.F1Score(threshold=0.5)
            else:
                compare_metric = keras.metrics.MeanAbsoluteError()
                weights = sample_weight

        if validation_data is None:
            x_metric, y_metric = x, y
        else:
            x_metric, y_metric, _ = validation_data

        def minimize(current, best):
            return current < best
        def maximize(current, best):
            return current > best

        compare_function = minimize
        if hasattr(compare_metric, "_direction"):
            if compare_metric._direction == 'up':
                compare_function = maximize

        predictions = model.predict(x_metric)

        if self._use_decoder_branch:
            y_metric = y_metric[0]

        if self._mode_enum == ModelType.REGRESSION:
            compare_metric.reset_state()
            compare_metric.update_state(y_true=y_metric, y_pred=predictions, sample_weight=weights)
            metric_result = compare_metric.result().numpy()
            if verbose_imbal > 0:
                print(f'Result of testing metric "{compare_metric.name}" for previous fit: {metric_result:.4f}')
            return metric_result, None, compare_function

        best_metric_result = None
        best_threshold = None
        best_threshold_min = None
        best_threshold_max = None

        for i in range(1, 10):
            compare_metric.reset_state()
            rounded_predictions = (predictions > i/10).astype(np.int32)

            compare_metric.update_state(y_true=y_metric, y_pred=rounded_predictions, sample_weight=weights)
            current_metric_result = compare_metric.result().numpy()
            current_threshold = i/10

            if current_metric_result.ndim > 0:
                current_metric_result = current_metric_result[0]

            if verbose_imbal > 1:
                print(f'Result of testing metric "{compare_metric.name}" with decision threshold {current_threshold}: {current_metric_result:.4f}')

            if best_metric_result is None or compare_function(current_metric_result, best_metric_result):
                best_metric_result = current_metric_result
                best_threshold_min = current_threshold
                best_threshold_max = current_threshold
                best_threshold = current_threshold

            elif np.isclose(current_metric_result, best_metric_result):
                best_threshold_min = min(best_threshold_min, current_threshold)
                best_threshold_max = max(best_threshold_max, current_threshold)
                best_threshold = (best_threshold_min + best_threshold_max) / 2.0

        if verbose_imbal > 0:
            print(f'Best decision threshold based on metric "{compare_metric.name}": {best_threshold}')

        return best_metric_result, best_threshold, compare_function

    def _store_metric_spec(self, metric):
        if metric is None:
            return None

        if isinstance(metric, str):
            return metric

        return keras.metrics.serialize(metric)

    def _retrieve_metric(self, metric_spec):
        if metric_spec is None:
            return None

        if isinstance(metric_spec, str):
            return keras.metrics.get(metric_spec)

        return keras.metrics.deserialize(metric_spec)

    def compile(
        self,
        generate_decoder_branch=False,
        representation_layer_index=-2,
        **kwargs
    ):
        """
        An extension of `TensorFlow's model.compile function <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_,
        which takes all the same parameters as the original function, along with some additional
        parameters used by the additional functionalities provided by this extended model class.

        Includes the ability to optionally generate a decoder branch extending from the provided model,
        which aids in achieving a better representation space, usually resulting in better performance
        on imbalanced data. This feature can be enabled by setting :code:`generate_decoder_branch` to :code:`True`.
        (see :doc:`imbal.util.generate_decoder </imbal/util/generate_decoder>` for more details).

        Note: :code:`generate_decoder_branch` is off by
        default due to its experimental nature, but it may be worth using if your model fits the
        anticipated structure.

        Args:
            generate_decoder_branch: Optional, default :code:`False`. Whether to generate a decoder
                branch for the purpose of training the model.
            representation_layer_index: Optional, default :code:`-2`. The layer from which the weights of all layers
                prior are frozen during the second stage of the decoupled training. Also, when
                :code:`generated_decoder_branch` is :code:`True`, the index of the layer from which the decoder branch
                in generated.
            **kwargs: Any keyword arguments accepted by `TensorFlow's model.compile function <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_

        Returns:
            None

        Example:

        .. code-block:: python

            inputs = keras.Input(shape=(28,28,1))
            x = layers.Conv2D(8, (3, 3), strides=(2, 2), activation='relu', padding='same')(inputs)
            x = layers.Conv2D(16, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
            x = layers.Flatten()(x)
            x = layers.Dense(16, activation='relu')(x)
            x = layers.Flatten()(x)
            output = layers.Dense(10, activation='softmax')(x)

            model =  imbal.classification.Model(inputs=inputs, outputs=output)

            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                metrics=["accuracy"]
            )
        """
        self._generate_decoder_branch = generate_decoder_branch
        self._representation_layer_index = representation_layer_index
        self._decoder_branch = None
        self._extended_model = None

        self._use_decoder_branch = self._generate_decoder_branch

        weighted_metrics = kwargs.get("weighted_metrics", None)
        metrics = kwargs.get("metrics", None)

        if weighted_metrics is not None and len(weighted_metrics) > 0:
            self._compare_metric_spec = self._store_metric_spec(weighted_metrics[0])
            self._compare_metric_uses_weights = True
        elif metrics is not None and len(metrics) > 0:
            self._compare_metric_spec = self._store_metric_spec(metrics[0])
            self._compare_metric_uses_weights = False
        else:
            self._compare_metric_spec = None
            self._compare_metric_uses_weights = False

        if self._generate_decoder_branch:
            imbal.util.generate_decoder(self)
            self._compile_for_decoder_branch(**kwargs)

        super().compile(**kwargs)

        self._compiled_optimizer = self.optimizer
        self.optimizer = copy.deepcopy(self._compiled_optimizer)

        self._compile_config = serialization_lib.SerializableDict(
            **kwargs,
            generate_decoder_branch=generate_decoder_branch,
            representation_layer_index=representation_layer_index
        )

    def compile_from_config(self, config):
        # Required to be overridden by Keras, however, the
        # implementation here is identical to Keras
        config = serialization_lib.deserialize_keras_object(config)
        self.compile(**config)
        if hasattr(self, "optimizer") and self.built:
            # Create optimizer variables.
            self.optimizer.build(self.trainable_variables)

    def get_compile_config(self):
        # Required to be overridden by Keras, however, the
        # implementation here is identical to Keras
        if self.compiled and hasattr(self, "_compile_config"):
            return self._compile_config.serialize()
        return None

    def _compile_for_decoder_branch(self, **kwargs):
        updated_compile_kwargs = kwargs.copy()
        model_loss = updated_compile_kwargs.get('loss', False)
        updated_compile_kwargs['loss'] = (
            [updated_compile_kwargs['loss'], mse_reconstruction_loss] if model_loss
            else mse_reconstruction_loss
        )

        model_metrics = kwargs.get('metrics', None)
        if model_metrics is not None:
            is_list_like = backend.tools.is_list_like(model_metrics[0])
            updated_compile_kwargs['metrics'] = (
                updated_compile_kwargs['metrics'] + [[]] if is_list_like
                else [updated_compile_kwargs['metrics']] + [[]]
            )

        weighted_model_metrics = kwargs.get('weighted_metrics', None)
        if weighted_model_metrics is not None:
            is_list_like = backend.tools.is_list_like(weighted_model_metrics[0])
            updated_compile_kwargs['weighted_metrics'] = (
                updated_compile_kwargs['weighted_metrics'] + [[]] if is_list_like
                else [updated_compile_kwargs['weighted_metrics']] + [[]]
            )

        self._extended_model.compile(**updated_compile_kwargs)

    def override_second_stage_fit_parameters(self, **kwargs):
        """
        Used to optionally override the parameters passed to the second stage of
        a decoupled fit. For instance, if you wanted to use a callback
        during the second stage of a decoupled fit, but not the first,
        you can call :code:`override_second_stage_fit_parameters` before
        calling the decoupled fit`, specifying a callback in
        :code:`override_second_stage_fit_parameters` but not in the fit call.

        Args:
            **kwargs: Any keyword arguments accepted by  `TensorFlow's model.fit function <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_

        Returns:
            None
        """
        self._second_stage_fit_kwargs = kwargs.copy()

    def _multi_weight(
        self,
        sample_weight=None,
        class_weight=None,
    ):
        if isinstance(sample_weight, list) or isinstance(sample_weight, np.ndarray):
            sample_weight = np.array(sample_weight)
            if sample_weight.ndim == 2:
                return True
        if sample_weight is None and (isinstance(class_weight, list) or isinstance(class_weight, np.ndarray)):
            class_weight = np.array(class_weight)
            if class_weight.ndim == 2:
                return True
        return False

    def _multi_weight_fit (
        self,
        model=None,
        x=None,
        y=None,
        sample_weight=None,
        validation_data=None,
        validation_split=None,
        candidate_evaluation_sample_weight=None,
        candidate_evaluation_class_weight=None,
        epochs=1,
        batch_size=32,
        shuffle=True,
        stratify_batches=True,
        verbose_imbal=1,
        class_weight=None,
        **kwargs
    ):

        weight_type = 'class weight' if class_weight is not None else 'sample weight'

        best_history = None
        best_model_weights = None
        best_weights_index = None
        best_metric_result = None
        best_threshold = None

        starting_model_weights = model.get_weights()

        if candidate_evaluation_sample_weight is not None:
            candidate_evaluation_weights = candidate_evaluation_sample_weight
        else:
            if self._mode_enum == ModelType.CLASSIFICATION:
                candidate_evaluation_weights = imbal.classification.generate_sample_weights(
                    y[0] if self._use_decoder_branch else y,
                    class_weight=candidate_evaluation_class_weight
                )
            else:
                raise RuntimeError(
                    'To perform a a fit with multiple weight candidates for regression, '
                    'some candidate evaluation weights must be specified'
                )

        def format_array_string(array):
            array = np.array(array)
            if array.shape[0] < 7:
                return str(array)
            else:
                return f'[{"  ".join([str(x) for x in array[:3]])}  ...  {"  ".join([str(x) for x in array[-3:]])}]'

        if self._use_decoder_branch and self._reconstruction_lambda is None:
            if verbose_imbal > 0:
                print(f'Determining reconstruction lambda...')
            self._reconstruction_lambda = self._determine_reconstruction_lambda(
                model,
                x=x,
                y=y,
                batch_size=None if stratify_batches else batch_size,
                shuffle=shuffle,
                **kwargs
            )

            if verbose_imbal > 0:
                print(f'Found reconstruction lambda {self._reconstruction_lambda:.4f}')

            config = model.get_compile_config()
            config['loss_weights'] = [1.0, float(self._reconstruction_lambda)]
            model.compile_from_config(config)

        for index, weights in enumerate(sample_weight):
            tf.keras.backend.clear_session()
            if stratify_batches:
                multi_fit_x, multi_fit_y, multi_fit_weights = self._stratify_data(x, y, weights, batch_size, shuffle)
            else:
                multi_fit_x, multi_fit_y, multi_fit_weights = x, y, weights

            current_kwargs = kwargs.copy()
            if 'callbacks' in kwargs:
                current_kwargs['callbacks'] = _clone_callbacks(kwargs['callbacks'])

            if verbose_imbal > 1:
                print(f'Performing fit on {weight_type} candidate at index {index}:\n{format_array_string(weights if weight_type == "sample weight" else class_weight[index])}')

            if validation_data is not None:
                x_val, y_val, w_val = validation_data
                current_val_data = (x_val, y_val, w_val[index])
            else:
                current_val_data = None

            if hasattr(self, 'compiled_metrics') and self.compiled_metrics:
                reset_fn = getattr(self.compiled_metrics, 'reset_states', None) or getattr(self.compiled_metrics,
                                                                                           'reset_state', None)
                if reset_fn:
                    reset_fn()

            history = keras.Model.fit(
                model,
                x=multi_fit_x,
                y=multi_fit_y,
                sample_weight=multi_fit_weights,
                validation_data=current_val_data,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                **current_kwargs
            )
            if verbose_imbal > 0:
                print(f'[{index+1}/{len(sample_weight)}] Fitted after {len(history.history.get("loss"))} epochs for {weight_type} candidate at index {index}')

            current_metric_result, current_threshold, compare_function = self._optimize_metric(
                self,
                x,
                y,
                candidate_evaluation_weights,
                current_val_data,
                verbose_imbal
            )


            if best_metric_result is None or compare_function(current_metric_result, best_metric_result):
                best_metric_result = current_metric_result
                best_threshold = current_threshold
                best_weights_index = index
                best_model_weights = model.get_weights()
                best_history = history

            if stratify_batches:
                del multi_fit_x

            model.set_weights(starting_model_weights)
            self._reset_optimizer(model)

        gc.collect()
        tf.keras.backend.clear_session()

        if verbose_imbal > 0:
            print(f'Restoring model weights from fit on {weight_type} candidate at index {best_weights_index}')
            if weight_type == 'class weight':
                print(f'Class weights of best fit: {format_array_string(class_weight[best_weights_index])}')

        if class_weight is not None:
            self.best_class_weights = class_weight[best_weights_index]
        else:
            self.best_sample_weights = sample_weight[best_weights_index]

        self.best_weight_index = best_weights_index
        self.best_decision_threshold = best_threshold
        model.set_weights(best_model_weights)

        return best_history

    def _prepare_training_data(
        self,
        x=None,
        y=None,
        class_weight=None,
        sample_density=None,
        sample_weight=None,
        validation_data=None,
        validation_densities=None,
        validation_split=None,
        shuffle=True,
        seed=None,
        require_weighting=False
    ):

        assert isinstance(x, (tf.Tensor, np.ndarray))
        assert isinstance(y, (tf.Tensor, np.ndarray))
        # Assumptions
        # - At least x is provided
        # - Data in x is a NumPy array/tensor

        # Order of operations:
        # - Make sure all training data is weighted
        # - Split validation data (if necessary)
        # - Make sure all validation data is weighted

        # Ensure some sample weights exist
        sample_weight = self._auto_compute_weights(
            y,
            sample_weight,
            class_weight,
            sample_density,
            require_weighting
        )


        # Split validation data if necessary
        if validation_split is not None and validation_data is None:
            (x, y, sample_weight), validation_data = imbal.util.backend.split(
                x, y, sample_weight,
                test_size=validation_split,
                shuffle=shuffle,
                seed=seed,
                mode=self._mode_enum
            )

        if sample_weight.ndim == 1:
            sample_weight = sample_weight[None, ...]

        # Ensure some validation weights exist
        if validation_data is not None:
            x_val, y_val, w_val = self._unpack_validation(validation_data)
            w_val = self._auto_compute_weights(
                y_val,
                w_val,
                class_weight,
                validation_densities,
                require_weighting
            )
            if w_val.ndim == 1:
                w_val = w_val[None, ...]

            w_val = verify_weight_scale(w_val, show_warning=False)
            assert w_val.shape[0] == sample_weight.shape[0]

            if self._use_decoder_branch:
                y_val = [y_val, x_val]

            validation_data = (x_val, y_val, w_val)

        sample_weight = verify_weight_scale(sample_weight, show_warning=False)
        if self._use_decoder_branch:
            y = [y, x]

        return (x, y, sample_weight), validation_data

    def _auto_compute_weights(
        self,
        labels,
        sample_weight,
        class_weight,
        sample_density,
        require_weighting
    ):
        if self._mode_enum == ModelType.CLASSIFICATION:
            if sample_weight is not None and class_weight is not None:
                warnings.warn('Both sample_weights and class_weight have been provided' +
                              'to balanced_fit. class_weight will be ignored.')

            if sample_weight is None and require_weighting:
                sample_weight = imbal.classification.generate_sample_weights(
                    labels,
                    class_weight=class_weight
                )
            elif sample_weight is None:
                sample_weight = np.ones(len(labels))
        else:
            if sample_weight is not None and sample_density is not None:
                warnings.warn('Both sample_weights and sample_densities have been provided' +
                              'to balanced_fit. sample_densities will be ignored.')

            if sample_weight is None and require_weighting:
                if sample_density is None:
                    raise ValueError('Must provide either sample_densities or sample_weights')
                sample_weight = self._mode_subpackage.generate_sample_weights(sample_density)
            elif sample_weight is None:
                sample_weight = np.ones(len(labels))
        return sample_weight


    def _unpack_validation(self, validation_data):
        if len(validation_data) == 2:
            x_val, y_val = validation_data
            w_val = None
        else:
            x_val, y_val, w_val = validation_data
        return x_val, y_val, w_val

    def _stratify_data(
        self,
        x,
        y,
        sample_weight,
        batch_size,
        shuffle
    ):
        if self._use_decoder_branch:
            x = backend.MultiDatasetWithBatching(
                x,
                y,
                sample_weights=sample_weight,
                batch_size=batch_size,
                shuffle=shuffle,
                multi_output=True,
                output_label_index=0,
                mode=self._mode_enum
            )
        else:
            x = backend.DatasetWithBatching(
                x,
                y,
                sample_weights=sample_weight,
                batch_size=batch_size,
                shuffle=shuffle,
                mode=self._mode_enum
            )
        return x, None, None



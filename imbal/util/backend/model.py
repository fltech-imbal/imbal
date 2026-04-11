import tensorflow as tf
import numpy as np
import keras, warnings
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.saving import serialization_lib

import imbal
import imbal.util.backend as backend
from imbal.util.backend.constants import ModelType
from imbal.util.backend.tools import verify_weight_scale
import copy

def mse_reconstruction_loss(y_true, y_pred):
    sq = tf.math.squared_difference(y_true, y_pred)
    axes = tf.range(1, tf.rank(sq))
    loss_per_example = tf.reduce_mean(sq, axis=axes)
    return loss_per_example

class Model(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.best_sample_weights = None
        self.best_class_weights = None
        self.best_metric_threshold = None

        self._generate_decoder_branch = False
        self._use_decoder_branch = False
        self._representation_layer_index = -2
        self._extended_model = None
        self._decoder_branch = None
        self._second_stage_fit_kwargs = {}
        self._mode_subpackage = None
        self._mode_enum = None

    def fit(
        self,
        x=None,
        y=None,
        sample_weight=None,
        validation_data=None,
        validation_split=None,
        batch_size=32,
        shuffle=True,
        stratify_batches=True,
        seed=None,
        verbose_imbal=1,
        **kwargs
    ):
        """
        An extension of `TensorFlow's model.fit function <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_
        that allows for data batches to be stratified if desired.

        Args:
            x: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                A NumPy array of data points, arranged as a column vector
            y: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                A NumPy array of labels, arranged as a row vector, column vector, or list of one-hot vectors.
            sample_weight: Optional, default :code:`None`. A list of sample weights. If specified,
                overrides :code:`class_weights`. Optionally, a 2D list of sample weights can be provided, in which case
                the model will be fit once all class weights provided, with the final model weights being set to the
                final weights from the fit with the best :code:`val_loss` (or :code:`loss` if no validation is specified).
            validation_data: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                The data used to validate the model during training.
                See `Tensorflow's model.fit documentation <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_.
            validation_split: Optional, default :code:`None`. A float value representing the proportion of the
                    provided training data to split off into a separate dataset used for model validation.
            batch_size: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                The batch size to use during training.
            shuffle: Optional, default :code:`True` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                Whether to shuffle the data before each epoch.
            stratify_batches: Optional, default :code:`True`. Whether to stratify data batch-wise during training.
                See :doc:`DatasetWithBatching </imbal/classification/dataset_with_batching>` for details.
                Only used when :code:`multi_output` is :code:`True`.
            verbose_imbal: Optional, default 1. Sets the verbosity level of :code:`imbal` related messages.
            **kwargs: Any additional keyword arguments accepted by `TensorFlow's model.fit function <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_

        Returns:
            A History object. Its History.history attribute is a record of training loss values and metrics values
            at successive epochs, as well as validation loss values and validation metrics values (if applicable).

        """

        if not self._mode_enum or not self._mode_subpackage:
            raise NotImplementedError

        return self._enforced_fit(
            x=x,
            y=y,
            class_weight=None,
            sample_density=None,
            sample_weight=sample_weight,
            validation_data=validation_data,
            validation_densities=None,
            validation_split=validation_split,
            batch_size=batch_size,
            shuffle=shuffle,
            stratify_batches=stratify_batches,
            verbose_imbal=verbose_imbal,
            seed=seed,
            require_weighting=False,
            **kwargs
        )

    def _balanced_fit(
        self,
        x=None,
        y=None,
        class_weight=None,
        sample_density=None,
        sample_weight=None,
        validation_data=None,
        validation_densities=None,
        validation_split=None,
        batch_size=32,
        shuffle=True,
        stratify_batches=False,
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
            validation_data=validation_data,
            validation_densities=validation_densities,
            validation_split=validation_split,
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
        validation_data=None,
        validation_densities=None,
        validation_split=None,
        epochs=1,
        batch_size=32,
        shuffle=True,
        stratify_batches=False,
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

        stage_one_history = self._enforced_fit(
            x=x,
            y=y,
            class_weight=None,
            sample_density=None,
            sample_weight=stage_one_sample_weights,
            validation_data=validation_data,
            validation_densities=None,
            validation_split=validation_split,
            batch_size=batch_size,
            shuffle=shuffle,
            stratify_batches=stratify_batches,
            verbose_imbal=verbose_imbal,
            seed=seed,
            require_weighting=False,
            epochs=stage_one_epochs,
            **kwargs
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
        second_stage_fit_kwargs['callbacks'] = None

        # Allow second stage overrides
        second_stage_fit_kwargs.update(self._second_stage_fit_kwargs)

        self._use_decoder_branch = False
        stage_two_history = self._enforced_fit(
            x=x,
            y=y,
            class_weight=class_weight,
            sample_density=sample_density,
            sample_weight=sample_weight,
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
        validation_data=None,
        validation_densities=None,
        validation_split=None,
        batch_size=32,
        shuffle=True,
        stratify_batches=False,
        verbose_imbal=1,
        seed=None,
        require_weighting=False,
        **kwargs
    ):
        (x, y, sample_weight), validation_data = self._prepare_training_data(
            x=x,
            y=y,
            class_weight=class_weight,
            sample_density=sample_density,
            sample_weight=sample_weight,
            validation_data=validation_data,
            validation_densities=validation_densities,
            validation_split=validation_split,
            shuffle=shuffle,
            seed=seed,
            require_weighting=require_weighting
        )

        training_model = self._extended_model if self._use_decoder_branch else self

        if sample_weight.shape[0] == 1:
            sample_weight = sample_weight.reshape(-1)
            if stratify_batches:
                x, y, sample_weight = self._stratify_data(x, y, sample_weight, batch_size, shuffle)

            history = keras.Model.fit(
                training_model,
                x=x,
                y=y,
                sample_weight=sample_weight,
                validation_data=validation_data,
                validation_split=validation_split,
                batch_size=None if stratify_batches else batch_size,
                shuffle=shuffle,
                **kwargs
            )
        else:
            history = self._multi_weight_fit(
                model=training_model,
                x=x,
                y=y,
                sample_weight=sample_weight,
                validation_data=validation_data,
                validation_split=validation_split,
                batch_size=batch_size,
                shuffle=shuffle,
                stratify_batches=stratify_batches,
                verbose_imbal=verbose_imbal,
                **kwargs
            )

        self._use_decoder_branch = self._generate_decoder_branch

        return history

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
        """
        self._generate_decoder_branch = generate_decoder_branch
        self._representation_layer_index = representation_layer_index
        self._decoder_branch = None
        self._extended_model = None

        self._use_decoder_branch = self._generate_decoder_branch

        if self._generate_decoder_branch:
            imbal.util.generate_decoder(self)
            self._compile_for_decoder_branch(**kwargs)

        super().compile(**kwargs)

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

        model_metrics = kwargs.get('metrics', [None])
        is_list_like = backend.tools.is_list_like(model_metrics[0])
        updated_compile_kwargs['metrics'] = (
            (
                updated_compile_kwargs['metrics'] + [['mse']] if is_list_like
                else [updated_compile_kwargs['metrics']] + [['mse']]
            ) if model_metrics
            else ['mse']
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

    def _multi_weight_fit(
        self,
        model=None,
        x=None,
        y=None,
        sample_weight=None,
        validation_data=None,
        validation_split=None,
        batch_size=32,
        shuffle=True,
        stratify_batches=True,
        verbose_imbal=1,
        class_weight=None,
        **kwargs
    ):

        weight_type = 'class weight' if class_weight is not None else 'sample weight'
        find_threshold = sample_weight is None

        best_loss = None
        best_history = None
        best_model_weights = None
        best_weights_index = None

        starting_model_weights = self.get_weights()

        def clone_callbacks(callbacks):
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

        for index, weights in enumerate(sample_weight):
            if stratify_batches:
                multi_fit_x, multi_fit_y, multi_fit_weights = self._stratify_data(x, y, weights, batch_size, shuffle)
            else:
                multi_fit_x, multi_fit_y, multi_fit_weights = x, y, sample_weight

            current_kwargs = kwargs.copy()
            if 'callbacks' in kwargs:
                current_kwargs['callbacks'] = clone_callbacks(kwargs['callbacks'])

            def format_array_string(array):
                array = np.array(array)
                if array.shape[0] < 7:
                    return str(array)
                else:
                    return f'[{"  ".join([str(x) for x in array[:3]])}  ...  {"  ".join([str(x) for x in array[-3:]])}]'
            if verbose_imbal > 1:
                print(f'Performing fit on {weight_type} candidate at index {index}:\n{format_array_string(weights)}')
            history = keras.Model.fit(
                model,
                x=multi_fit_x,
                y=multi_fit_y,
                sample_weight=multi_fit_weights,
                validation_data=validation_data,
                validation_split=validation_split,
                batch_size=batch_size,
                shuffle=shuffle,
                **current_kwargs
            )
            if verbose_imbal > 0:
                print(f'[{index+1}/{len(sample_weight)}] Fitted after {len(history.history.get("loss"))} epochs for {weight_type} candidate at index {index}')

            loss_metric = history.history.get('val_loss', None)
            if loss_metric is None:
                loss_metric = history.history.get('loss', None)

            best_loss_index = np.argmin(loss_metric)
            best_loss_of_run = loss_metric[best_loss_index]

            if best_loss is None or best_loss_of_run < best_loss:
                best_loss = best_loss_of_run
                best_history = history
                del best_model_weights  # free previous
                best_model_weights = self.get_weights()
                best_weights_index = index

            self.set_weights(starting_model_weights)

        if verbose_imbal > 0:
            print(f'Restoring model weights from fit on {weight_type} candidate at index {best_weights_index}')

        if class_weight is not None:
            self.best_class_weights = class_weight[best_weights_index]
        else:
            self.best_sample_weights = sample_weight[best_weights_index]

        self.set_weights(best_model_weights)

        if find_threshold and len(self.compiled_metrics._metrics) > 0:
            compare_metric = self.compiled_metrics._metrics[0]
            best_result = None
            best_threshold = None
            predictions = self.predict(kwargs['x'], kwargs['y'])
            for i in range(1, 10):
                compare_metric.reset_state()
                rounded_predictions = (predictions > i/10).astype(np.int32)
                compare_metric.update_state(y_true=kwargs['y'], y_pred=rounded_predictions)
                if best_result is None or compare_metric.result()[0] > best_result:
                    best_result = compare_metric.result()[0]
                    best_threshold = i/10
            self.best_threshold = best_threshold

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
        if sample_weight.ndim == 1:
            sample_weight = sample_weight[None, ...]

        # Split validation data if necessary
        if validation_split is not None and validation_data is None:
            (x, y, sample_weight), validation_data = imbal.util.backend.split(
                x, y, sample_weight,
                test_size=validation_split,
                shuffle=shuffle,
                seed=seed,
                mode=self._mode_enum
            )

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
            if w_val.ndims == 1:
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
                warnings.warn('Both sample_weights and class_weights have been provided' +
                              'to balanced_fit. class_weights will be ignored.')

            if sample_weight is None and require_weighting:
                sample_weight = self._mode_subpackage.generate_sample_weights(
                    labels,
                    class_weights=class_weight
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
            x = self._mode_subpackage.DatasetWithBatching(
                x,
                y,
                sample_weights=sample_weight,
                batch_size=batch_size,
                shuffle=shuffle
            )
        return x, None, None



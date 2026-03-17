import tensorflow as tf
import numpy as np
import keras, warnings
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.saving import serialization_lib
import imbal
import imbal.util.backend as backend
from imbal.util.backend.constants import ModelType
from imbal.util.backend.tools import verify_weight_scale

def mse_reconstruction_loss(y_true, y_pred):
    sq = tf.math.squared_difference(y_true, y_pred)
    axes = tf.range(1, tf.rank(sq))
    loss_per_example = tf.reduce_mean(sq, axis=axes)
    return loss_per_example

class Model(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        stratify_batches=False,
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
            **kwargs: Any additional keyword arguments accepted by `TensorFlow's model.fit function <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_

        Returns:
            A History object. Its History.history attribute is a record of training loss values and metrics values
            at successive epochs, as well as validation loss values and validation metrics values (if applicable).

        """

        """
        SAME (1)
        """
        if not self._mode_enum or not self._mode_subpackage:
            raise NotImplementedError
        """
        SAME (2)
        """
        if self._multi_weight(sample_weight, None):
            return self._multi_weight_fit(
                self.fit,
                x=x,
                y=y,
                sample_weight=sample_weight,
                validation_data=validation_data,
                validation_split=validation_split,
                batch_size=batch_size,
                shuffle=shuffle,
                stratify_batches=stratify_batches,
                **kwargs
            )


        if stratify_batches or self._use_decoder_branch:
            x, y, sample_weight, stratify_batches = self._x_y_weight_split_data(x, y, sample_weight, stratify_batches)

        """
        NEAR SAME (3)
        """
        if sample_weight is None and not isinstance(x, tf.data.Dataset) and not isinstance(x, keras.utils.PyDataset):
            sample_weight = np.ones(x.shape[0])

        sample_weight = verify_weight_scale(sample_weight)

        """
        SAME (4)
        """
        if validation_split and validation_data is None:
            (x, y, sample_weight), (val_x, val_y, val_sample_weight) = self._mode_subpackage.split(
                x,
                y,
                sample_weights=sample_weight,
                test_size=validation_split,
            )
            sample_weight = verify_weight_scale(sample_weight, show_warning=False)
            val_sample_weight = verify_weight_scale(val_sample_weight, show_warning=False)
            validation_data = (val_x, val_y, val_sample_weight)

        """
        NEAR SAME (5)
        """
        if validation_data is not None:
            if isinstance(validation_data, self._mode_subpackage.DatasetWithBatching):
                val_x, val_y, val_sample_weight = validation_data.unpack()
            else:
                (
                    val_x,
                    val_y,
                    val_sample_weight,
                ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)

            if self._use_decoder_branch:
                val_y = [val_y, val_x]
            val_sample_weight = verify_weight_scale(val_sample_weight)
            validation_data = (val_x, val_y, val_sample_weight)

        training_model = self
        if self._use_decoder_branch:
            training_model = self._extended_model
            y = [y, x]

        if stratify_batches:
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
            y = None
            sample_weight = None

        history = keras.Model.fit(
            training_model,
            x=x,
            y=y,
            sample_weight=sample_weight,
            validation_split=validation_split,
            validation_data=validation_data,
            batch_size=None if stratify_batches else batch_size,
            shuffle=shuffle,
            **kwargs
        )

        self._use_decoder_branch = self._generate_decoder_branch

        return history

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
        **kwargs
    ):
        """
        Ignore
        """

        """
        SAME (1)
        """
        if not self._mode_enum or not self._mode_subpackage:
            raise NotImplementedError

        """
        SAME (2)
        """
        if stratify_batches or self._use_decoder_branch:
            x, y, sample_weight, stratify_batches = self._x_y_weight_split_data(x, y, sample_weight, stratify_batches)

        """
        NEAR SAME (3)
        """
        if sample_weight is None and not isinstance(x, tf.data.Dataset) and not isinstance(x, keras.utils.PyDataset):
            sample_weight = self._auto_compute_weights(y, sample_weight, class_weight, sample_density)

        """
        SAME (4)
        """
        if validation_split and validation_data is None:
            (x, y, sample_weight), (val_x, val_y, val_sample_weight) = self._mode_subpackage.split(
                x,
                y,
                sample_weights=sample_weight,
                test_size=validation_split,
            )
            sample_weight = verify_weight_scale(sample_weight, show_warning=False)
            val_sample_weight = verify_weight_scale(val_sample_weight, show_warning=False)
            validation_data = (val_x, val_y, val_sample_weight)

        """
        NEAR SAME (5)
        """
        if validation_data is not None:
            if isinstance(validation_data, self._mode_subpackage.DatasetWithBatching):
                val_x, val_y, val_sample_weight = validation_data.unpack()
            else:
                (
                    val_x,
                    val_y,
                    val_sample_weight,
                ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)

            if val_sample_weight is None:
                if validation_densities is not None:
                    val_sample_weight = imbal.regression.generate_sample_weights(validation_densities)
                else:
                    combined_y = np.concatenate((y, val_y), axis=0)
                    combined_weights = self._auto_compute_weights(combined_y, None, class_weight, None)
                    sample_weight = combined_weights[:y.shape[0]]
                    sample_weight = verify_weight_scale(sample_weight, show_warning=False)
                    val_sample_weight = combined_weights[y.shape[0]:]
                    val_sample_weight = verify_weight_scale(val_sample_weight, show_warning=False)
                validation_data = (val_x, val_y, val_sample_weight)

        return self.fit(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_data=validation_data,
            validation_split=validation_split,
            stratify_batches=stratify_batches,
            **kwargs
        )

    def _x_y_weight_split_data(
        self,
        x,
        y,
        sample_weight,
        stratify_batches
    ):
        data = []
        labels = []
        weights = []
        if isinstance(x, tf.data.Dataset):
            if stratify_batches:
                warnings.warn("In order to utilize batch stratification, data must be passed as a NumPy"
                              "array, array-like, tensor, or PyDataset. Batch stratification has been"
                              "disabled for this fit.")
                stratify_batches=False
            if self._use_decoder_branch:
                warnings.warn("In order to utilize decoder branch generation, data must be passed as a NumPy"
                              "array, array-like, tensor, or PyDataset. Decoder branch generation has been"
                              "disabled for this fit.")
                self._use_decoder_branch = False
        elif isinstance(x, keras.utils.PyDataset):
            for i in range(x.num_batches):
                batch = x[i]
                if len(batch) == 2:
                    data.append(batch[0])
                    labels.append(batch[1])
                    weights.append(np.ones(len(batch[0])))
                elif len(batch) == 3:
                    data.append(batch[0])
                    labels.append(batch[1])
                    weights.append(batch[2])
                else:
                    raise RuntimeError("PyDataset could not be split into data, labels, and weights")
            x = np.concatenate(data)
            y = np.concatenate(labels)
            sample_weight = np.concatenate(weights)
        return x, y, sample_weight, stratify_batches

    def _auto_compute_weights(
            self,
            labels,
            sample_weight,
            class_weight,
            sample_density
    ):
        if self._mode_enum == ModelType.CLASSIFICATION:
            if sample_weight is not None and class_weight is not None:
                warnings.warn('Both sample_weights and class_weights have been provided' +
                              'to balanced_fit. class_weights will be ignored.')
            if sample_weight is None:
                sample_weight = self._mode_subpackage.generate_sample_weights(
                    labels,
                    class_weights=class_weight
                )
        else:
            if sample_weight is not None and sample_density is not None:
                warnings.warn('Both sample_weights and sample_densities have been provided' +
                              'to balanced_fit. sample_densities will be ignored.')
            if sample_weight is None:
                if sample_density is None:
                    raise ValueError('Must provide either sample_densities or sample_weights')
                sample_weight = self._mode_subpackage.generate_sample_weights(sample_density)
        return sample_weight

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
        **kwargs
    ):
        """
        Ignore
        """
        """
        SAME (1)
        """
        if not self._mode_enum or not self._mode_subpackage:
            raise NotImplementedError

        if isinstance(epochs, tuple):
            stage_one_epochs, stage_two_epochs = epochs
        else:
            stage_one_epochs = epochs
            stage_two_epochs = None

        """
        SAME (4)
        """
        if validation_split and validation_data is None:
            (x, y, sample_weight), (val_x, val_y, val_sample_weight) = self._mode_subpackage.split(
                x,
                y,
                sample_weights=sample_weight,
                test_size=validation_split,
            )
            sample_weight = verify_weight_scale(sample_weight, show_warning=False)
            val_sample_weight = verify_weight_scale(val_sample_weight, show_warning=False)
            validation_data = (val_x, val_y, val_sample_weight)

        stage_one_x = x
        stage_one_y = y

        training_model = self
        if self._use_decoder_branch:
            training_model = self._extended_model
            stage_one_y = [y, x]

        val_x, stage_two_val_y, stage_two_val_sample_weight = None, None, None

        if validation_data is not None:
            if isinstance(validation_data, self._mode_subpackage.DatasetWithBatching):
                val_x, val_y, val_sample_weight = validation_data.unpack()
            else:
                (
                    val_x,
                    val_y,
                    val_sample_weight,
                ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)
                stage_two_val_y = val_y
            if self._use_decoder_branch:
                val_y = [val_y, val_x]
            stage_two_val_sample_weight = verify_weight_scale(val_sample_weight)
            val_sample_weight = np.ones(val_sample_weight.shape)
            validation_data = (val_x, val_y, val_sample_weight)

        stage_one_sample_weights = np.ones(x.shape[0])

        if stratify_batches:
            if self._use_decoder_branch:
                stage_one_x = backend.MultiDatasetWithBatching(
                    x,
                    stage_one_y,
                    sample_weights=stage_one_sample_weights,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    multi_output=True,
                    output_label_index=0,
                    mode=self._mode_enum
                )
            else:
                stage_one_x = self._mode_subpackage.DatasetWithBatching(
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
        second_stage_fit_kwargs['epochs'] = len(stage_one_history.epoch) if stage_two_epochs is None else stage_two_epochs
        second_stage_fit_kwargs['sample_weight'] = sample_weight
        second_stage_fit_kwargs['validation_data'] = None if validation_data is None else (val_x, stage_two_val_y, stage_two_val_sample_weight)
        second_stage_fit_kwargs['callbacks'] = None
        second_stage_fit_kwargs['batch_size'] = batch_size
        second_stage_fit_kwargs['shuffle'] = shuffle
        second_stage_fit_kwargs['validation_split'] = validation_split
        second_stage_fit_kwargs['stratify_batches'] = stratify_batches

        # Allow second stage overrides
        second_stage_fit_kwargs.update(self._second_stage_fit_kwargs)

        self._use_decoder_branch = False
        stage_two_history = self._balanced_fit(
            x=x,
            y=y,
            class_weight=class_weight,
            validation_densities=validation_densities,
            sample_density=sample_density,
            **second_stage_fit_kwargs
        )
        self._use_decoder_branch = self._generate_decoder_branch
        if self._generate_decoder_branch:
            self._extended_model.trainable = True

        return stage_one_history, stage_two_history  # In the future, potentially only second stage history is returned

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
        fit_function,
        **kwargs
    ):
        sample_weight = kwargs.pop('sample_weight', None)
        class_weight = kwargs.pop('class_weight', None)

        iterate_over = sample_weight if sample_weight is not None else class_weight
        iterate_over = np.array(iterate_over)

        best_loss = None
        best_history = None
        best_model_weights = None

        starting_model_weights = self.get_weights()

        serialized_kwargs = serialization_lib.SerializableDict(
            callbacks=kwargs.pop('callbacks', None),
        )

        for weights in iterate_over:
            current_kwargs = kwargs.copy()
            if sample_weight is not None:
                current_kwargs['sample_weight'] = weights
            else:
                current_kwargs['class_weight'] = weights

            current_kwargs.update(serialization_lib.deserialize_keras_object(serialized_kwargs))

            history = fit_function(
                **kwargs
            )

            loss_metric = history.history.get('val_loss', None)
            if loss_metric is None:
                loss_metric = history.history.get('loss', None)

            best_loss_index = np.argmin(loss_metric)
            best_loss_of_run = loss_metric[best_loss_index]

            if best_loss is None or best_loss_of_run < best_loss:
                best_loss = best_loss_of_run
                best_history = history
                best_model_weights = self.get_weights()

            self.set_weights(starting_model_weights)

        self.set_weights(best_model_weights)
        return best_history


import tensorflow as tf
import numpy as np
import keras, warnings, math
from keras.src.saving import serialization_lib
from keras.src.trainers.data_adapters import data_adapter_utils

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
        self._serialized_compile_kwargs = None
        self._generate_decoder_branch = False
        self._use_decoder_branch = False
        self._stratify_batches = False
        self._perform_batch_stratification = False
        self._representation_layer_index = -2
        self._extended_model = None
        self._decoder_branch = None
        self._second_stage_compile_kwargs = {}
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
            **kwargs:

        Returns:

        """
        if not self._mode_enum or not self._mode_subpackage:
            raise NotImplementedError

        if self._perform_batch_stratification or self._use_decoder_branch:
            x, y, sample_weight = self._x_y_weight_split_data(x, y, sample_weight)

        if sample_weight is None and not isinstance(x, tf.data.Dataset) and not isinstance(x, keras.utils.PyDataset):
            sample_weight = np.ones(x.shape[0])

        sample_weight = verify_weight_scale(sample_weight)

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

        training_model = self
        if self._use_decoder_branch:
            training_model = self._extended_model
            y = [y, x]

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

        if self._perform_batch_stratification:
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
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )

        self._perform_batch_stratification = self._stratify_batches
        self._use_decoder_branch = self._generate_decoder_branch

        return history

    def balanced_fit(
        self,
        x=None,
        y=None,
        class_weight=None,
        sample_density=None,
        sample_weight=None,
        validation_data=None,
        validation_split=None,
        batch_size=32,
        shuffle=True,
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
        if not self._mode_enum or not self._mode_subpackage:
            raise NotImplementedError

        if self._perform_batch_stratification or self._use_decoder_branch:
            x, y, sample_weight = self._x_y_weight_split_data(x, y, sample_weight)

        if sample_weight is None and not isinstance(x, tf.data.Dataset) and not isinstance(x, keras.utils.PyDataset):
            sample_weight = self._auto_compute_weights(y, sample_weight, class_weight, sample_density)

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
                combined_y = np.concatenate((y, val_y), axis=0)
                combined_weights = self._auto_compute_weights(combined_y, None, class_weight, None)
                sample_weight = combined_weights[:y.shape[0]]
                sample_weight = verify_weight_scale(sample_weight, show_warning=False)
                val_sample_weight = combined_weights[y.shape[0]:]
                val_sample_weight = verify_weight_scale(val_sample_weight, show_warning=False)
                validation_data = (val_x, val_y, val_sample_weight)

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

        return self.fit(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_data=validation_data,
            validation_split=validation_split,
            **kwargs
        )

    def _x_y_weight_split_data(
        self,
        x,
        y,
        sample_weight,
    ):
        data = []
        labels = []
        weights = []
        if isinstance(x, tf.data.Dataset):
            if self._perform_batch_stratification:
                warnings.warn("In order to utilize batch stratification, data must be passed as a NumPy"
                              "array, array-like, tensor, or PyDataset. Batch stratification has been"
                              "disabled for this fit.")
                self._perform_batch_stratification = False
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
        return x, y, sample_weight

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

    def decoupled_fit(
        self,
        x=None,
        y=None,
        sample_weight=None,
        validation_data=None,
        validation_split=None,
        epochs=1,
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
        if not self._mode_enum or not self._mode_subpackage:
            raise NotImplementedError

        training_model = self

        if isinstance(epochs, tuple):
            first_train_epochs, second_train_epochs = epochs
        else:
            first_train_epochs = epochs
            second_train_epochs = math.ceil(epochs / 2)

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

        stage_two_y = y
        if self._use_decoder_branch:
            training_model = self._extended_model
            y = [y, x]

        val_x, stage_two_val_y, val_sample_weight = None, None, None
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
                stage_two_val_y = val_y
                val_y = [val_y, val_x]
            val_sample_weight = verify_weight_scale(val_sample_weight)
            validation_data = (val_x, val_y, val_sample_weight)

        stage_one_history = training_model.fit(
            x=x,
            y=y,
            sample_weight=sample_weight,
            validation_data=validation_data,
            validation_split=validation_split,
            epochs=first_train_epochs,
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

        second_stage_compile_parameters = serialization_lib.deserialize_keras_object(self._serialized_compile_kwargs)
        second_stage_compile_parameters.update(self._second_stage_compile_kwargs)
        second_stage_fit_kwargs = kwargs.copy()
        second_stage_fit_kwargs['epochs'] = second_train_epochs
        second_stage_fit_kwargs['sample_weight'] = sample_weight
        second_stage_fit_kwargs['validation_data'] = None if validation_data is None else (val_x, stage_two_val_y, val_sample_weight)
        second_stage_fit_kwargs['validation_split'] = validation_split
        second_stage_fit_kwargs.update(self._second_stage_fit_kwargs)

        model_clone = keras.models.clone_model(self)
        model_clone.set_weights(self.get_weights())
        model_clone.compile(**second_stage_compile_parameters)

        self._use_decoder_branch = False
        self._perform_batch_stratification = False
        self.trainable = True
        stage_two_history = model_clone.balanced_fit(
            x=x,
            y=stage_two_y,
            **second_stage_fit_kwargs
        )
        self.set_weights(model_clone.get_weights())
        self._use_decoder_branch = self._generate_decoder_branch
        self._perform_batch_stratification = self._stratify_batches


        if self._generate_decoder_branch:
            self._extended_model.trainable = True

        return stage_one_history, stage_two_history  # In the future, potentially only second stage history is returned

    def compile(
        self,
        stratify_batches=False,
        generate_decoder_branch=False,
        representation_layer_index=-2,
        **kwargs
    ):
        """
        TODO: compile description

        Args:
            stratify_batches:
            generate_decoder_branch:
            representation_layer_index:
            **kwargs:

        Returns:

        """
        self._serialized_compile_kwargs = serialization_lib.serialize_keras_object(kwargs)
        self._generate_decoder_branch = generate_decoder_branch
        self._representation_layer_index = representation_layer_index
        self._stratify_batches = stratify_batches
        self._decoder_branch = None
        self._extended_model = None

        self._perform_batch_stratification = self._stratify_batches
        self._use_decoder_branch = self._generate_decoder_branch

        if self._generate_decoder_branch:
            self._generate_decoder()
            self._compile_for_decoder_branch(**kwargs)

        super().compile(**kwargs)
        self._compile_config = serialization_lib.SerializableDict(
            stratify_batches=stratify_batches,
            generate_decoder_branch=generate_decoder_branch,
            representation_layer_index=representation_layer_index,
            **kwargs
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
        deserialized_compile_kwargs = serialization_lib.deserialize_keras_object(self._serialized_compile_kwargs)
        model_loss = deserialized_compile_kwargs.get('loss', False)
        updated_compile_kwargs['loss'] = (
            [updated_compile_kwargs['loss'], mse_reconstruction_loss] if model_loss
            else mse_reconstruction_loss
        )

        model_metrics = deserialized_compile_kwargs.get('metrics', None)
        is_list_like = backend.tools.is_list_like(model_metrics[0])
        updated_compile_kwargs['metrics'] = (
            (
                updated_compile_kwargs['metrics'] + [['mse']] if is_list_like
                else [updated_compile_kwargs['metrics']] + [['mse']]
            ) if model_metrics
            else ['mse']
        )

        self._extended_model.compile(**updated_compile_kwargs)

    def override_second_stage_compile_parameters(self, **kwargs):
        """
        TODO: description

        Args:
            **kwargs:

        Returns:

        """
        self._second_stage_compile_kwargs = kwargs.copy()

    def override_second_stage_fit_parameters(self, **kwargs):
        """
        TODO: description

        Args:
            **kwargs:

        Returns:

        """
        self._second_stage_fit_kwargs = kwargs.copy()

    def _generate_decoder(self):
        """
        This function attempts to extend a provided model, using a simple algorithm
        which uses the ordering of the layers leading up to the specified representation
        layer to generate a decoder branch that reconstructs the model input.

        The algorithm for generating the structure of the decoder is as follows:

        - Divide the structure layers of the model leading up to the representation layer
          into blocks, where each trainable layer indicates the first layer in a new block
        - Reverse the order of the blocks, while maintaining the order of the layers within
          each block (i.e. [[A, B, C], [D, E, F]] goes to  [[D, E, F], [A, B, C]])
        - Remove all non-trainable layers from the last block of the decoder. This helps to
          ensure that the input data can be reconstructed without loss of generality due to
          the range of activation functions.

        Additional methods are employed to help ensure consistent change of shape between
        each layer of the decoder, resulting in a final output shape that is the same
        as the input shape of the model (ex. Conv2D layers are converted to Conv2DTranspose layers,
        and vice versa).

        For this function to work well, the provided model should be a linear, and the
        specified representation layer should be a Keras Flatten layer. Additionally,
        the representation layer should be relatively deep in the model, though it does not necessarily have to be
        at the end of the model. If you model already contains a Flatten layer (ex. 2D image data is flattened
        to a 1D vector), it is likely best to specify that layer as the representation layer.

        Returns:
            A tuple of the form :code:`(model, layers)`, where :code:`model` is the decoder-extended
            model, and :code:`layers` is a list of layer generated by this function.
        """

        representation_layer_index = backend.tools.positive_model_layer_index(self, self._representation_layer_index)
        reverse_model = self.layers[:representation_layer_index][::-1]

        # Determine AE blocks
        ae_blocks = []
        last_block_end_index = 0
        for index, layer in enumerate(reverse_model):
            if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
                ae_blocks.append(reverse_model[last_block_end_index:index + 1][::-1])
                last_block_end_index = index + 1
        # Exclude input layer
        if last_block_end_index != len(reverse_model) - 1:
            ae_blocks.append(reverse_model[last_block_end_index:-1][::-1])

        print('\n----- FOUND BLOCKS -----\n')  # Debug, delete later
        for block in ae_blocks:
            print('----- BLOCK -----')
            for layer in block:
                print(f'\t{layer}, {layer.get_config()}')  # Debug, delete later
        print('\n\n')

        # Perform per-layer conversions (i.e. Conv2D to Conv2DTranspose)
        # within each block
        ae_branch_blocks = []
        for block_index, block in enumerate(ae_blocks):
            current_input_shape = block[-1].output.shape[1:]
            current_ae_block = []
            reshape_layer = keras.layers.Reshape(current_input_shape,
                                                 name=f'imbal_auto_generated_ae_safeguard_reshape_block_{block_index}')
            current_ae_block.append(reshape_layer)
            print('----- BLOCK -----')  # Debug, delete later
            for layer_index, layer in enumerate(block):
                new_layer = None
                config = layer.get_config()
                config['name'] = f'imbal_auto_generated_ae_block_{block_index}_layer_{layer_index}'
                if isinstance(layer, keras.layers.Conv2D):
                    layer_shape_change = layer.input.shape[-1] / layer.output.shape[-1]
                    config.pop('groups', None)
                    config['filters'] = round(config['filters'] * layer_shape_change)
                    new_layer = keras.layers.Conv2DTranspose(**config)
                elif isinstance(layer, keras.layers.Conv2DTranspose):
                    layer_shape_change = layer.input.shape[-1] / layer.output.shape[-1]
                    config['filters'] = round(config['filters'] * layer_shape_change)
                    new_layer = keras.layers.Conv2D(**config)
                elif isinstance(layer, keras.layers.MaxPooling2D):
                    config.pop('groups', None)
                    config['strides'] = layer.strides
                    config.pop('pool_size', None)
                    config['kernel_size'] = layer.pool_size
                    config['filters'] = round(block[layer_index-1].get_config()['filters'])
                    new_layer = keras.layers.Conv2DTranspose(**config)
                elif isinstance(layer, keras.layers.Dense):
                    units = layer.input.shape[-1]
                    config['units'] = units
                    new_layer = keras.layers.Dense(**config)

                # Failsafe for non-trainable layers
                elif not (hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer')):
                    new_layer = type(layer).from_config(config)

                # Raise exception if layer could not be converted
                if new_layer is None:
                    raise RuntimeError(f'Unable to perform AE conversion of layer {layer}')

                print(f'\t{new_layer}')  # Debug, delete later
                print(f'\t\t{new_layer.get_config()}')  # Debug, delete later

                current_ae_block.append(new_layer)
            ae_branch_blocks.append(current_ae_block)

        # For better results, last block should only be made on trainable layers (activation
        # and normalization layers can sometimes prevent reaching the goal reconstruction)
        refined_last_block = []
        for layer in ae_branch_blocks[-1]:
            if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer')\
                    or isinstance(layer, keras.layers.MaxPooling2D):
                refined_last_block.append(layer)
        ae_branch_blocks[-1] = refined_last_block

        print('\n----- AE CONVERSION -----\n')  # Debug, delete later
        for block in ae_branch_blocks:
            print('----- BLOCK -----')
            for layer in block:
                print(f'\t{layer}, {layer.get_config()}')  # Debug, delete later

        # Connect final layer structure
        ae_layer_list = [layer for block in ae_branch_blocks for layer in block]
        last_layer = self.layers[representation_layer_index]
        for layer in ae_layer_list:
            print(last_layer.output.shape)
            layer(last_layer.output)
            last_layer = layer

        if not(hasattr(self, 'inputs') and hasattr(self, 'outputs')):
            raise RuntimeError('Model\'s "inputs" and "outputs" fields are not set.')

        self._extended_model = self.__class__(inputs=self.inputs, outputs=self.outputs + [ae_layer_list[-1].output])
        self._decoder_branch = ae_layer_list



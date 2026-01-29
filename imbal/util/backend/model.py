import tensorflow as tf
import keras
from keras.src.saving import serialization_lib

import imbal.util.backend.tools as tools

def mse_reconstruction_loss(y_true, y_pred):
    sq = tf.math.squared_difference(y_true, y_pred)
    axes = tf.range(1, tf.rank(sq))
    loss_per_example = tf.reduce_mean(sq, axis=axes)
    return loss_per_example

class Model(keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compile_kwargs = None

        self._generate_decoder_branch = False
        self._stratify_batches = False
        self._representation_layer_index = -2

        self._extended_model = None
        self._decoder_branch = None

        self._second_stage_compile_kwargs = {}
        self._second_stage_fit_kwargs = {}

    def fit(
        self,
        *args,
        **kwargs
    ):
        has_overridden_method = self.__class__.fit != Model.fit
        if not has_overridden_method:
            raise NotImplementedError("Function 'fit' is not implemented by subclass")

    def balanced_fit(
        self,
        *args,
        **kwargs
    ):
        has_overridden_method = self.__class__.balanced_fit != Model.balanced_fit
        if not has_overridden_method:
            raise NotImplementedError("Function 'balanced_fit' is not implemented by subclass")

    def decoupled_fit(
        self,
        *args,
        **kwargs
    ):
        has_overridden_method = self.__class__.decoupled_fit != Model.decoupled_fit
        if not has_overridden_method:
            raise NotImplementedError("Function 'decoupled_fit' is not implemented by subclass")

    def compile(
        self,
        stratify_batches=False,
        generate_decoder_branch=False,
        representation_layer_index=None,
        **kwargs
    ):
        self._compile_kwargs = kwargs
        self._generate_decoder_branch = generate_decoder_branch
        self._representation_layer_index = representation_layer_index
        self._stratify_batches = stratify_batches
        self._decoder_branch = None
        self._extended_model = None

        if self._generate_decoder_branch:
            self.generate_decoder_branch()
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

        model_loss = self._compile_kwargs.get('loss', False)
        updated_compile_kwargs['loss'] = (
            [updated_compile_kwargs['loss'], mse_reconstruction_loss] if model_loss
            else mse_reconstruction_loss
        )

        model_metrics = self._compile_kwargs.get('metrics', None)
        is_list_like = tools.is_list_like(model_metrics[0])
        updated_compile_kwargs['metrics'] = (
            (
                updated_compile_kwargs['metrics'] + [['mse']] if is_list_like
                else [updated_compile_kwargs['metrics']] + [['mse']]
            ) if model_metrics
            else ['mse']
        )

        self._extended_model.compile(**updated_compile_kwargs)

    def override_second_stage_compile_parameters(self, **kwargs):
        self._second_stage_compile_kwargs = kwargs.copy()

    def override_second_stage_fit_parameters(self, **kwargs):
        self._second_stage_fit_kwargs = kwargs.copy()

    def generate_decoder_branch(self):
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

        representation_layer_index = tools.positive_model_layer_index(self, self._representation_layer_index)
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



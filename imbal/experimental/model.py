import tensorflow.keras as keras
from keras.src.saving import serialization_lib

from imbal.util.backend.fit.generate_decoder_branch import generate_decoder_branch as generate_branch
from imbal.util.backend.fit.generate_decoder_branch import mse_reconstruction_loss

class Model(keras.Model):

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self._compile_kwargs = None

    def fit(
        self,
        stratify_batches=False,
        generate_decoder_branch=False,
        representation_layer_index=-2,
        **kwargs
    ):
        super(Model, self).fit(**kwargs)

    def balanced_fit(
        self,
        x=None,
        y=None,
        stratify_batches=False,
        generate_decoder_branch=False,
        representation_layer_index=-2,
        **kwargs
    ):
        balanced_fit_compile_parameters = self._compile_kwargs.copy()

        if generate_decoder_branch:
            compiling_model, _ = generate_branch(self, representation_layer_index)

            model_loss = self._compile_kwargs.get('loss', None)
            if model_loss is None:
                balanced_fit_compile_parameters['loss'] = mse_reconstruction_loss
            else:
                balanced_fit_compile_parameters['loss'] = [balanced_fit_compile_parameters['loss'], mse_reconstruction_loss]

            model_metrics = self._compile_kwargs.get('metrics', None)
            if model_metrics is None:
                balanced_fit_compile_parameters['metrics'] = ['mse']
            else:
                if isinstance(model_metrics[0], list) or isinstance(model_metrics[0], tuple):
                    balanced_fit_compile_parameters['metrics'] = balanced_fit_compile_parameters['metrics'] + [['mse']]
                else:
                    balanced_fit_compile_parameters['metrics'] = [balanced_fit_compile_parameters['metrics']] + [['mse']]

            y = [y, x]

        has_branch = isinstance(y, list) and len(y) == 2

        dataset = x
        if mode == 'classification':
            if sample_weights is not None and class_weights is not None:
                warnings.warn('Both sample_weights and class_weights have been provided' +
                              'to balanced_fit. class_weights will be ignored.')
            if sample_weights is None:
                if has_branch:
                    sample_weights = classification.generate_sample_weights(y[0], class_weights=class_weights)
                else:
                    sample_weights = classification.generate_sample_weights(y, class_weights=class_weights)
            if stratify_batches:
                if has_branch:
                    dataset = imbal.util.backend.MultiDatasetWithBatching(
                        x,
                        y,
                        sample_weights=sample_weights,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        multi_output=True,
                        output_label_index=0,
                        mode='classification'
                    )
                else:
                    dataset = classification.DatasetWithBatching(
                        x,
                        y,
                        sample_weights=sample_weights,
                        batch_size=batch_size,
                        shuffle=shuffle
                    )
        else:
            if sample_weights is not None and sample_densities is not None:
                warnings.warn('Both sample_weights and sample_densities have been provided' +
                              'to balanced_fit. sample_densities will be ignored.')
            if sample_weights is None:
                if sample_densities is None:
                    raise ValueError('Must provide either sample_densities or sample_weights')
                sample_weights = regression.generate_sample_weights(sample_densities)

            if stratify_batches:
                if has_branch:
                    dataset = imbal.util.backend.MultiDatasetWithBatching(
                        x,
                        y,
                        sample_weights=sample_weights,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        multi_output=True,
                        output_label_index=0,
                        mode='regression'
                    )
                else:
                    dataset = regression.DatasetWithBatching(
                        x,
                        y,
                        sample_weights=sample_weights,
                        batch_size=batch_size,
                        shuffle=shuffle
                    )

        compiling_model.compile(**extended_parameters)

        compiling_model.fit(
            x=dataset,
            y=None if stratify_batches else y,
            sample_weight=None if stratify_batches else sample_weights,
            epochs=epochs,
            validation_data=validation_data
        )

        model.compile(**compile_parameters)

    def decoupled_fit(
        self,
        stratify_batches=False,
        generate_decoder_branch=False,
        representation_layer_index=-2,
        **kwargs
    ):
        return

    def compile(self, **kwargs):
        self._compile_kwargs = kwargs
        super(Model, self).compile(**kwargs)

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


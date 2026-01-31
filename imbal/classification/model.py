import warnings, math, keras
import numpy as np

import imbal.classification as classification
from imbal.util import get_representation_layer_index
import imbal.util.backend as backend
import imbal.util.backend.tools as tools

class Model(backend.Model):

    def fit(
        self,
        x=None,
        y=None,
        sample_weight=None,
        batch_size=32,
        shuffle=True,
        **kwargs
    ):
        training_model = self
        if self._generate_decoder_branch:
            training_model = self._extended_model
            y = [y, x]
            if kwargs.get('validation_data', None) is not None:
                val_x, val_y = kwargs['validation_data']
                kwargs['validation_data'] = (val_x, [val_y, val_x])

        if (abs(np.sum(sample_weight) - np.sum(sample_weight.shape))) > 10e-3:
            sample_weight *= np.sum(sample_weight.shape)/np.sum(sample_weight)
            warnings.warn('Provided weights do not sum to the number of samples. The provided weights'
                          'will be automatically normalized to sum to the number of samples.')

        dataset = x
        if self._stratify_batches:
            if self._generate_decoder_branch:
                dataset = backend.MultiDatasetWithBatching(
                    x,
                    y,
                    sample_weights=sample_weight,
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
                    sample_weights=sample_weight,
                    batch_size=batch_size,
                    shuffle=shuffle
                )

        return keras.Model.fit(
            training_model,
            x=dataset,
            y=None if self._stratify_batches else y,
            sample_weight=None if self._stratify_batches else sample_weight,
            **kwargs
        )

    def balanced_fit(
        self,
        x=None,
        y=None,
        class_weight=None,
        sample_weight=None,
        batch_size=32,
        shuffle=True,
        **kwargs
    ):
        if sample_weight is not None and class_weight is not None:
            warnings.warn('Both sample_weights and class_weights have been provided' +
                          'to balanced_fit. class_weights will be ignored.')
        if sample_weight is None:
            sample_weight = classification.generate_sample_weights(
                y,
                class_weights=class_weight
            )
            class_weight = None

        return self.fit(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )

    def decoupled_fit(
        self,
        x=None,
        y=None,
        epochs=1,
        *args,
        **kwargs
    ):
        self.trainable = True
        training_model = self

        if isinstance(epochs, tuple):
            first_train_epochs, second_train_epochs = epochs
        else:
            first_train_epochs = epochs
            second_train_epochs = math.ceil(epochs / 2)

        if self._generate_decoder_branch:
            training_model = self._extended_model
            y = [y, x]
            if kwargs.get('validation_data', None) is not None:
                val_x, val_y = kwargs['validation_data']
                kwargs['validation_data'] = (val_x, [val_y, val_x])

        stage_one_history = training_model.fit(
            x,
            y,
            epochs=first_train_epochs,
            **kwargs
        )

        representation_layer_index = tools.positive_model_layer_index(training_model, self._representation_layer_index)

        found_layer, found_index = get_representation_layer_index(
            training_model,
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
        if self._generate_decoder_branch:
            for layer in self._decoder_branch:
                layer.trainable = False

        stage_two_history = training_model.balanced_fit(
            x,
            y,
            epochs=second_train_epochs,
            **kwargs
        )

        self.trainable = True
        if self._generate_decoder_branch:
            self._extended_model.trainable = True

        return stage_one_history, stage_two_history # In the future, potentially only second stage history is returned

    def cRT_fit(
        self,
        *args,
        **kwargs
    ):
        return self.decoupled_fit(*args, **kwargs)
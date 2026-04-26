import imbal
import imbal.util.backend as backend
from imbal.util.backend.constants import ModelType

class Model(backend.Model):
    """
    Package location: :code:`imbal.classification.Model`

    An extension of `Keras's Model class <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_ that
    can perform alternative types of model fits which aim to address common issues when applying
    machine learning algorithms to imbalanced/long-tail data.

    This :code:`imbal.classification.Model` class is intended for use only on classification problems.

    For a comparison of how these fit methods perform compared to a standard fit on imbalanced/long-tail
    data, see the information at the bottom of this page.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode_enum = ModelType.CLASSIFICATION
        self._mode_subpackage = imbal.classification

    def balanced_fit(
        self,
        x=None,
        y=None,
        class_weight=None,
        sample_weight=None,
        validation_data=None,
        validation_split=None,
        batch_size=32,
        shuffle=True,
        stratify_batches=False,
        verbose_imbal=1,
        **kwargs
    ):
        """
        Performs a class-balanced fit, where by default, each class is equally weighted during
        the model fitting process.

        Args:
            x: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                A NumPy array of data points, arranged as a column vector
            y: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                A NumPy array of labels, arranged as a row vector, column vector, or list of one-hot vectors.
            class_weight: Optional, default :code:`None`. If left as default, equal class weighting is used.
                A list of class weights, or a dictionary mapping class
                labels to class weights. Optionally, a 2D list of class weights can be provided, in which case
                the model will be fit once using each class weight list provided, with the final model weights being set to the
                model weights from the fit with the best :code:`val_loss` (or :code:`loss` if no validation is specified).
            sample_weight: Optional, default :code:`None`. If set, overrides behavior of :code:`class_weights`. A list of sample weights. If specified,
                overrides :code:`class_weights`. Optionally, a 2D list of sample weights can be provided, in which case
                the model will be fit once using each sample weight list provided, with the final model weights being set to the
                final model weights from the fit with the best :code:`val_loss` (or :code:`loss` if no validation is specified).
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

        Example:

        .. code-block:: python

            # Assume MNIST data is already loaded in '(x_train, y_train), (x_test, y_test)'

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

            model.balanced_fit(
                x_train,
                y_train,
                validation_split=0.2
            )
        """
        return super()._balanced_fit(
            x=x,
            y=y,
            class_weight=class_weight,
            sample_weight=sample_weight,
            validation_data=validation_data,
            validation_split=validation_split,
            batch_size=batch_size,
            shuffle=shuffle,
            stratify_batches=stratify_batches,
            verbose_imbal=verbose_imbal,
            **kwargs
        )

    def cRT_fit(
        self,
        x=None,
        y=None,
        class_weight=None,
        sample_weight=None,
        validation_data=None,
        validation_split=None,
        epochs=1,
        batch_size=32,
        shuffle=True,
        stratify_batches=False,
        verbose_imbal=1,
        **kwargs
    ):
        """
        Performs a decoupled fit, based on the
        classifier re-training (cRT) method as described in
        `this paper by Kang et al. (ICLR 2020) <https://arxiv.org/abs/1910.09217>`_.

        Args:
            x: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                A Numpy array of data points, arranged as a column vector
            y: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                A Numpy array of labels, arranged as a row vector, column vector, or list of one-hot vectors.
            class_weight: Optional, default :code:`None`. A list of class weights, or a dictionary mapping class
                labels to class weights. Optionally, a 2D list of class weights can be provided, in which case
                the model will be fit once all class weights provided, with the final model weights being set to the
                final weights from the fit with the best :code:`val_loss` (or :code:`loss` if no validation is specified).
            sample_weight: Optional, default :code:`None`. A list of sample weights. If specified,
                overrides :code:`class_weights`. Optionally, a 2D list of sample weights can be provided, in which case
                the model will be fit once all class weights provided, with the final model weights being set to the
                final weights from the fit with the best :code:`val_loss` (or :code:`loss` if no validation is specified).
            validation_data: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                The data used to validate the model during training.
                See `Tensorflow's model.fit documentation <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.
            validation_split: Optional, default :code:`None`. A float value representing the proportion of the
                provided training data to split off into a separate dataset used for model validation.
            epochs: Optional, default :code:`1` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                The number of epochs to train for. If an :code:`int`,
                the provided number of epochs will be used during the first stage, and halved for the second stage.
                If a tuple or list of length 2, the value in the first index will be used as the number of
                epochs in the first stage of training, and the second index for the number of epochs
                in the second stage of training.
            batch_size: Optional, default :code:`32` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                The batch size to use during training.
            shuffle: Optional, default :code:`True` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                Whether to shuffle the data before each epoch.
            stratify_batches: Optional, default :code:`True`. Whether to stratify data batch-wise during training.
                See :doc:`DatasetWithBatching </imbal/classification/dataset_with_batching>` for details.
            **kwargs: Any additional keyword arguments accepted by `TensorFlow's model.fit function <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_

        Returns:
            A tuple (s1, s2) of History objects. Its History.history attribute is a record of training loss values
            and metrics values at successive epochs, as well as validation loss values and validation metrics
            values (if applicable).

        Example:

        .. code-block:: python
            # Assume MNIST data is already loaded in '(x_train, y_train), (x_test, y_test)'

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
                metrics=["accuracy"],
                representation_layer_index=-2
            )

            model.cRT_fit(
                x_train,
                y_train,
                validation_split=0.2
            )

        """
        return self._decoupled_fit(
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            validation_data=validation_data,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            stratify_batches=stratify_batches,
            verbose_imbal=verbose_imbal,
            **kwargs
        )
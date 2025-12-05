import imbal

def balanced_fit(
    model,
    x=None,
    y=None,
    class_weights=None,
    sample_weights=None,
    compile_parameters=None,
    batch_size=32,
    epochs=1,
    validation_data=None,
    shuffle=True,
):
    """
    Performs a class-balanced fit, where each class is equally weighted during
    the model fitting process.

    Args:
        model: The model to perform the decoupled fit on.
        x: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
            A NumPy array of data points, arranged as a column vector
        y: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
            A NumPy array of labels, arranged as a row vector, column vector, or list of one-hot vectors.
        class_weights: Optional, default :code:`None`. A list of class weights, or a dictionary mapping class
            labels to class weights.
        sample_weights: Optional, default :code:`None`. A list of sample weights. If specified,
            overrides :code:`class_weights`.
        compile_parameters: Optional, default :code:`None`. A :doc:`TFModelCompileParameters </imbal/helpers/tf_model_compile_parameters>`
            object, or a dictionary mapping `Tensorflow model.compile parameters <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_
            to their corresponding values. If set to :code:`None`, the default `model.compile <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_
            parameters will be used.
        batch_size: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
            The batch size to use during training.
        epochs: Optional, default :code:`1` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
            The number of epochs to train for. If an :code:`int`,
            the provided number of epochs will be used during the first stage, and halved for the second stage.
            If a tuple or list of length 2, the value in the first index will be used as the number of
            epochs in the first stage of training, and the second index for the number of epochs
            in the second stage of training.
        validation_data: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
            The data used to validate the model during training.
            See `Tensorflow's model.fit documentation <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_.
        shuffle: Optional, default :code:`True` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
            Whether to shuffle the data before each epoch.

    Returns:
        :code:`None`

    Example:

    .. code-block:: python

        >>> # Assume MNIST data is already loaded in '(x_train, y_train), (x_test, y_test)'
        >>> # and that a model is already created in 'model'

        >>> import imbal
        >>> import keras

        >>> parameters = imbal.regression.compile_parameters(
        >>>     loss="categorical_crossentropy",
        >>>     optimizer=keras.optimizers.Adam(),
        >>>     metrics=["accuracy"]
        >>> )

        >>> imbal.regression.balanced_fit(
        >>>     model,
        >>>     x_train,
        >>>     y_train,
        >>>     compile_parameters=parameters,
        >>>     epochs=10,
        >>>     batch_size=512
        >>> )

    """
    imbal.util.backend.balanced_fit(
        model,
        x=x,
        y=y,
        compile_parameters=compile_parameters,
        sample_weights=sample_weights,
        class_weights=class_weights,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        shuffle=shuffle,
        mode='classification'
    )
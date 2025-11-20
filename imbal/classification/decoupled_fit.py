import imbal

def decoupled_fit(
    model,
    x=None,
    y=None,
    compile_parameters=None,
    stage_one_compile_parameters=None,
    stage_two_compile_parameters=None,
    batch_size=32,
    epochs=1,
    validation_data=None,
    shuffle=True,
    representation_layer_index=-2,
    aed_for_representation=True
):
    """
    Performs a decoupled fit on the provided model, as described in
    `this paper by Kang et al. <https://arxiv.org/abs/1910.09217>`_.


    Args:
        model: The model to perform the decoupled fit on.
        x: Optional, default :code:`None`. A NumPy array of data points, arranged as a column vector
        y: Optional, default :code:`None`. A NumPy array of labels, arranged as a row vector, column vector, or list of one-hot vectors.
        compile_parameters: Optional, default :code:`None`. A :doc:`TFModelCompileParameters </imbal/helpers/tf_model_compile_parameters>`
            object, or a dictionary mapping `Tensorflow model.compile parameters <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_
            to their corresponding values. Will be used in place of :code:`stage_one_compile_parameters`
            and :code:`stage_two_compile_parameters`, if one or neither is provided.
        stage_one_compile_parameters: Optional, default :code:`None`. A :doc:`TFModelCompileParameters </imbal/helpers/tf_model_compile_parameters>`
            object, or a dictionary mapping `Tensorflow model.compile parameters <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_
            to their corresponding values. These parameters are used to compile the model
            during the first (representation learning) stage of the decoupled fit.
        stage_two_compile_parameters: Optional, default :code:`None`. A :doc:`TFModelCompileParameters </imbal/helpers/tf_model_compile_parameters>`
            object, or a dictionary mapping `Tensorflow model.compile parameters <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_
            to their corresponding values. These parameters are used to compile the model
            during the second (classifier learning) stage of the decoupled fit.
        batch_size: Optional, default :code:`32`. The batch size to use during training.
        epochs: Optional, default :code:`1`. The number of epochs to train for. If an :code:`int`,
            the provided number of epochs will be used during the first and second stages of training.
            If a tuple or list of length 2, the value in the first index will be used as the number of
            epochs in the first stage of training, and the second index for the number of epochs
            in the second stage of training.
        validation_data: Optional, default :code:`None`. The data used to validate the model during training.
            See `Tensorflow's model.fit documentation <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.
        shuffle: Optional, default :code:`True`. Whether to shuffle the data before each epoch.
        representation_layer_index: Optional, default :code:`-2`. The index of the representation layer
            in the provided model's :code:`model.layers` list.
        aed_for_representation: Optional, default :code:`True`. TODO.

    Returns:
        :code:`None`

    Example:

    .. code-block:: python

        >>> # Assume MNIST data is already loaded in '(x_train, y_train), (x_test, y_test)'
        >>> # and that a model is already created in 'model'

        >>> import imbal
        >>> import keras

        >>> parameters = imbal.classification.compile_parameters(
        >>>     loss="categorical_crossentropy",
        >>>     optimizer=keras.optimizers.Adam(),
        >>>     metrics=["accuracy"]
        >>> )

        >>> imbal.classification.decoupled_fit(
        >>>     model,
        >>>     x_train,
        >>>     y_train,
        >>>     compile_parameters=parameters,
        >>>     epochs=10,
        >>>     batch_size=512
        >>> )

    """
    imbal.util.decoupled_fit(
        model,
        x=x,
        y=y,
        compile_parameters=compile_parameters,
        stage_one_compile_parameters=stage_one_compile_parameters,
        stage_two_compile_parameters=stage_two_compile_parameters,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        shuffle=shuffle,
        representation_layer_index=representation_layer_index,
        aed_for_representation=aed_for_representation,
        mode='classification'
    )
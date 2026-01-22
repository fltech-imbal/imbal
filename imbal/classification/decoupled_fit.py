import imbal

def cRT_fit(
    model,
    x=None,
    y=None,
    class_weights=None,
    sample_weights=None,
    compile_parameters=None,
    stage_one_compile_parameters=None,
    stage_two_compile_parameters=None,
    validation_data=None,
    epochs=1,
    batch_size=32,
    shuffle=True,
    stratify_batches=True,
    generate_decoder_branch=False,
    representation_layer_index=-2,
):
    """
    Performs a decoupled fit on the provided model, based on the
    classifier re-training (cRT) method as described in
    `this paper by Kang et al. (ICLR 2020) <https://arxiv.org/abs/1910.09217>`_.

    Includes the ability to optionally generate a decoder branch extending from the provided model,
    which aids in achieving a better representation space, usually resulting in better performance
    on imbalanced data. This feature, enabled using :code:`generate_decoder_branch`, is off by
    default due to its experimental nature, but we recommend using it if your model fits the
    anticipated structure (see :doc:`imbal.util.backend.generate_decoder_branch </imbal/util/backend/generate_decoder_branch>` for more details).

    Args:
        model: The model to perform the decoupled fit on.
        x: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
            A Numpy array of data points, arranged as a column vector
        y: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
            A Numpy array of labels, arranged as a row vector, column vector, or list of one-hot vectors.
        class_weights: Optional, default :code:`None`. A list of class weights, or a dictionary mapping class
            labels to class weights.
        sample_weights: Optional, default :code:`None`. A Numpy array of sample weights.
        compile_parameters: Optional, default :code:`None`. A :doc:`TFModelCompileParameters </imbal/util/model_compile_parameters>`
            object, or a dictionary mapping `Tensorflow model.compile parameters <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_
            to their corresponding values. If set to :code:`None`, the default `model.compile <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_
            parameters will be used.
        stage_one_compile_parameters: Optional, default :code:`None`. A :doc:`TFModelCompileParameters </imbal/helpers/tf_model_compile_parameters>`
            object, or a dictionary mapping `Tensorflow model.compile parameters <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_
            to their corresponding values. These parameters are used to compile the model
            during the first (representation learning) stage of the decoupled fit. If set to :code:`None`,
            will be overriden by the value of :code:`compile_parameters`.
        stage_two_compile_parameters: Optional, default :code:`None`. A :doc:`ModelCompileParameters </imbal/helpers/tf_model_compile_parameters>`
            object, or a dictionary mapping `Tensorflow model.compile parameters <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_
            to their corresponding values. These parameters are used to compile the model
            during the second (classifier learning) stage of the decoupled fit.  If set to :code:`None`,
            will be overriden by the value of :code:`compile_parameters`.
        validation_data: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
            The data used to validate the model during training.
            See `Tensorflow's model.fit documentation <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.
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
        generate_decoder_branch: Optional, default :code:`False`. When set to :code:`True`, an extended version of
            the provided model containing a decoder branch is generated and used for training, often yielding
            better training results (see :doc:`Comparison of Fit Methods </imbal/classification/comparison_of_fit_methods>`).
            Decoder generation is experimental and may not always be possible depending on model structure.
        representation_layer_index: Optional, default :code:`-2`. The layer from which the weights of all layers prior are frozen during
            the stage of the decoupled training. Also, when :code:`generated_decoder_branch` is :code:`True`, the index of
            the layer from which the decoder branch in generated.


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

        >>> # For example, the models representation layer is the fourth from the end...
        >>> imbal.classification.cRT_fit(
        >>>     model,
        >>>     x_train,
        >>>     y_train,
        >>>     compile_parameters=parameters,
        >>>     epochs=10,
        >>>     batch_size=512
        >>>     representation_layer_index=-4,
        >>> )

    """
    imbal.util.backend.decoupled_fit(
        model,
        x=x,
        y=y,
        compile_parameters=compile_parameters,
        stage_one_compile_parameters=stage_one_compile_parameters,
        stage_two_compile_parameters=stage_two_compile_parameters,
        sample_weights=sample_weights,
        class_weights=class_weights,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        shuffle=shuffle,
        representation_layer_index=representation_layer_index,
        generate_decoder_branch=generate_decoder_branch,
        mode='classification',
        stratify_batches=stratify_batches
    )
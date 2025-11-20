class TFModelCompileParameters:
    """
    A Simple object which can store the parameters to be passed to Tensorflow's
    :code:`model.compile` function.
    """
    def __init__(self, optimizer='rmsprop', loss=None, metrics=None, **kwargs):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        """
        Returns the parameters stored in this object, as a dictionary.

        Returns: a :code:`dict` object.
        """
        return self.__dict__

def compile_parameters(
    optimizer='rmsprop',
    loss=None,
    metrics=None,
    **kwargs
):
    """
    A wrapper function to construct a :doc:`TFModelCompileParameters </imbal/helpers/tf_model_compile_parameters>`
    object. Implemented to help mimic Tensorflow's process for compling models.

    Args:
        optimizer: Optional, default :code:`'rmsprop'`. The optimizer to use during model compilation.
        loss: Optional, default :code:`None`. The loss function to use during model compilation.
        metrics: Optional, default :code:`None`. The metrics to use during model compilation.
        **kwargs: Any additional parameters to include during model compilation.

    Returns:
        A :doc:`TFModelCompileParameters </imbal/helpers/tf_model_compile_parameters>` object
        containing the passed parameters.
    """
    return TFModelCompileParameters(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
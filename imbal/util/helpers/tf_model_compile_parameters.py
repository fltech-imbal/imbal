class TFModelCompileParameters:
    def __init__(self, optimizer='rmsprop', loss=None, metrics=None, **kwargs):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return self.__dict__

def compile_parameters(
    optimizer='rmsprop',
    loss=None,
    metrics=None,
    **kwargs
):
    return TFModelCompileParameters(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
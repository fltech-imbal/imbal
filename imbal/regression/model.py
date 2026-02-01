import imbal
import imbal.util.backend as backend
from imbal.util.backend.constants import ModelType

class Model(backend.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode_enum = ModelType.REGRESSION
        self._mode_subpackage = imbal.regression

    def rRT_fit(
        self,
        *args,
        **kwargs
    ):
        return self.decoupled_fit(*args, **kwargs)
import imbal
import imbal.util.backend as backend
from imbal.util.backend.constants import ModelType

class Model(backend.Model):
    """
    Classification model
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode_enum = ModelType.CLASSIFICATION
        self._mode_subpackage = imbal.classification

    def cRT_fit(
        self,
        *args,
        **kwargs
    ):
        """
        :code:`crt_fit` is an alias of :code:`decoupled_fit` (see
        :code:`decoupled_fit` documentation on this page).

        Returns:
            A 2-tuple of :code:`History` objects, one from each stage
            of the decoupled fit.

        """
        return self.decoupled_fit(*args, **kwargs)
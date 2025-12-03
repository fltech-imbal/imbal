class KDEFitParameters:
    """
    A simple object which can store the parameters to be passed to
    :code:`imbal.regression.fit_kde`.
    """
    def __init__(
        self,
        fit_method='kl_divergence',
        average_samples_per_bin=100,
        bin_count=None,
        steps_per_bin=10,
        num_candidates=10,
        tolerance=0,
        padding_factor=0.01
    ):
        self.fit_method = fit_method
        self.average_samples_per_bin = average_samples_per_bin
        self.bin_count = bin_count
        self.steps_per_bin = steps_per_bin
        self.num_candidates = num_candidates
        self.tolerance = tolerance
        self.padding_factor = padding_factor

    def to_dict(self):
        """
        Returns the parameters stored in this object, as a dictionary.

        Returns: a :code:`dict` object.
        """
        return self.__dict__

def wrap_kde_fit_parameters(
    fit_method='kl_divergence',
    average_samples_per_bin=100,
    bin_count=None,
    steps_per_bin=10,
    num_candidates=10,
    tolerance=0,
    padding_factor=0.01
):
    """
    A wrapper function to construct a :doc:`ModelCompileParameters </imbal/util/model_compile_parameters>`
    object. Implemented to help mimic Tensorflow's process for compling models.

    For additional details about individual parameters, see :doc:`imbal.regression.fit_kde </imbal/regression/fit_kde>`.

    Args:
        fit_method: Optional, default :code:`'kl_divergence'`. A string indicating the method to use to
            determine bandwidth.
        average_samples_per_bin: Optional, default :code:`100`. Determines the
            number of bins used for histogram-based KDE approximation by the number of datapoints.
        bin_count: Optional, default :code:`None`. The number of bins that
            should be used for the histogram-based KDE approximation.
            If set, overrides :code:`average_samples_per_bin`.
        steps_per_bin: Optional, default :code:`10`. Determines the number of
            steps per bin that should be used for KDE optimizations.
        num_candidates: Optional, default :code:`10`. For iterative approach only. Determines
            the number of candidates to check during each round of beam search.
        tolerance: Optional, default :code:`0`. For iterative approach only. Determines the tolerance
            within which the iterative heuristic can be considered close to 0, allowing iteration to end.
        padding_factor: Optional, default :code:`0.01`. Used to add a small padding to
            the data range used for binning for the histogram. There are some instances where many datapoints in
            a dataset fall on the maximum or minimum.

    Returns:
        A :doc:`KDEFitParameters </imbal/helpers/kde_fit_parameters>` object
        containing the passed parameters.
    """
    return KDEFitParameters(
        fit_method=fit_method,
        average_samples_per_bin=average_samples_per_bin,
        bin_count=bin_count,
        steps_per_bin=steps_per_bin,
        num_candidates=num_candidates,
        tolerance=tolerance,
        padding_factor=padding_factor,
    )
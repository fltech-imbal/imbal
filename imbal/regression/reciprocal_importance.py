import numpy as np
from imbal.util.backend import tools

def reciprocal_importance(
    sample_densities,
    alpha=1,
    steps=10
):
    """

    Calculates the reciprocal importance for the provided sample
    densities. Can optionally be used to generate the reciprocal
    importance of the densities for a spread of alpha values.
    The resulting weights will always sum to n, where n is the
    number of density values provided.

    Args:
        sample_densities: A NumPy array of sample densities.
        alpha: Optional, default :code:`1`. If set to a float, the provided value
            will be used to calculate the reciprocal importance weight. If set to
            a tuple :code:`(start, end)`, alpha values between :code:`start` and
            :code:`end` will be interpolated and used to calculate reciprocal
            importance weights. If set to a list, the values within the list
            will be used as the alpha values for calculating reciprocal importance.
        steps: Optional, default :code:`10`. Only used if :code:`alpha` is set to a
            tuple. The number of steps to perform when interpolating between the
            range of alpha values provided.

    Returns:
        If :code:`alpha` is a singular float, a list of weights generated from the
        reciprocal importance function. If :`alpha` is a tuple, a 2D array of weights,
        where each row corresponds to the weights generated from the reciprocal importance
        function for a particular alpha value.

    Example:

    .. code::

        from imbal.regression import reciprocal_importance
        import numpy as np

        densities = np.array([10, 1, 0.1, 0.01])
        weights = reciprocal_importance(densities, alpha=(0,1), steps=3)

        # The resulting value of weights is shown below
        # [[1.         1.         1.         1.        ]
        #  [0.08736475 0.27627161 0.87364754 2.7627161 ]
        #  [0.00360036 0.0360036  0.360036   3.60036004]]
    """
    if isinstance(alpha, tuple):
        sample_densities = np.array(sample_densities).reshape(-1)
        alphas = np.arange(steps)/(steps-1) * (alpha[1] - alpha[0]) + alpha[0]
        alphas = alphas.reshape(-1, 1)
    elif isinstance(alpha, (list, np.ndarray)):
        alphas = np.array([[x] for x in alpha])
    else:
        alphas = np.array([[alpha]])

    weights = 1 / (np.power(sample_densities, alphas))
    weights = tools.verify_weight_scale(weights, show_warning=False, axis=1)
    return np.squeeze(weights)
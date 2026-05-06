import numpy as np
from scipy.stats import gaussian_kde
from .label_to_importance_map import map_labels_to_importance_weights

def mdi_importance(densities, alpha=1):
    min_density = np.min(densities)
    max_density = np.max(densities)

    # Normalize densities to [0, 1] range with max_pdf + epsilon mapping to 1
    normalized_densities = np.divide(densities - min_density, max_density - min_density + 1e-3)

    # Calculate reweighting factors using power function
    # y = [1- x^(alpha)]^(1/alpha) where x is normalized density
    importance_weights = np.power(1 - np.power(normalized_densities, alpha), 1.0 / alpha)

    # Normalize importance weights to sum to 1 using numpy ops
    importance_weights = np.divide(importance_weights, np.sum(importance_weights))
    importance_weights = importance_weights / np.sum(importance_weights)
    importance_weights = importance_weights.reshape(-1)

    return importance_weights

class MDI:
    """
    Class for generating importance weights based on MDI.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        alpha = 2,
        bandwidth = .07,
        epsilon = 1e-3
    ) -> None:
        """
        Initialize the MDI class.

        The reweighting uses importance_weight = [1- density^(alpha)]^(1/alpha) where density is normalized density between 0 and 1.

        :param features: Training instances.
        :param labels: Training labels.
        :param bandwidth: bandwidth for the KDE.
        :param alpha: importance weight coefficient for power function
        :param epsilon: small value added to max_pdf for normalization
        """

        # Create training data
        self.features = features
        self.labels = labels

        # Create KDE and get PDF values
        self.kde = gaussian_kde(self.labels, bw_method=bandwidth)
        self.densities = self.kde.evaluate(self.labels)

        # Smooth the density values for high outliers
        self.densities = smooth_blip(self.labels, self.densities)

        self.min_density = np.min(self.densities)
        self.max_density = np.max(self.densities)

        # Normalize densities to [0, 1] range with max_pdf + epsilon mapping to 1
        normalized_densities = np.divide(self.densities, self.max_density + epsilon)

        # Calculate reweighting factors using power function
        # y = [1- x^(alpha)]^(1/alpha) where x is normalized density
        self.importance_weights = np.power(1 - np.power(normalized_densities, alpha), 1.0 / alpha)

        # Normalize importance weights to sum to 1 using numpy ops
        self.importance_weights = np.divide(self.importance_weights, np.sum(self.importance_weights))

        # Create mapping dictionary
        self.label_importance_map = map_labels_to_importance_weights(
            self.labels, self.importance_weights)


def smooth_blip(x: np.ndarray, density: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    """
    Smooths density values above threshold to the minimum density value found above threshold.
    This prevents jumps in density values for extreme outliers.

    Parameters:
    - x: np.ndarray
      The input points where density is evaluated.
    - density: np.ndarray
      The original KDE density values.
    - threshold: float
      The threshold above which to smooth density values.

    Returns:
    - np.ndarray
      Density values with high outliers smoothed to minimum density above threshold.
    """
    smoothed_density = density.copy()

    # Get indices above threshold
    mask = x >= threshold
    if not np.any(mask):
        return smoothed_density

    # Find minimum density value above threshold
    min_density_above = np.min(density[mask])

    # Set all densities above threshold to minimum value
    smoothed_density[mask] = min_density_above

    return smoothed_density
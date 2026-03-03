import numpy as np
import imbal.util.backend.tools as tools

def dense_weight(
    sample_densities,
    alpha=1,
    steps=10,
    epsilon=0.01,
):
    if isinstance(alpha, tuple):
        sample_densities = np.array(sample_densities).reshape(-1)
        alphas = np.arange(steps) / (steps - 1) * (alpha[1] - alpha[0]) + alpha[0]
        alphas = alphas.reshape(-1, 1)
    elif isinstance(alpha, (list, np.ndarray)):
        alphas = np.array([[x for x in alpha]])
    else:
        alphas = np.array([[alpha]])

    sample_densities = sample_densities.reshape(1, -1)
    alphas = alphas.reshape(-1, 1)

    density_min, density_max = np.min(sample_densities), np.max(sample_densities)
    normalized_densities = (sample_densities - density_min) / (density_max - density_min)
    weights = np.clip(1 - alphas * normalized_densities, epsilon, None)
    weights = tools.verify_weight_scale(weights, show_warning=False, axis=1)
    return np.squeeze(weights)
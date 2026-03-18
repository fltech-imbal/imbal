import numpy as np

def denseweight_sample_weights(densities, alpha=0.6, eps=1e-12):
    d = np.asarray(densities, dtype=np.float64)

    # p'(y): scale density to [0, 1]
    d_max = np.max(d)
    if d_max <= 0:
        # degenerate fallback: all same weight
        return np.ones_like(d, dtype=np.float32)

    p_prime = d / d_max

    # f'_w(alpha,y) = max(1 - alpha * p'(y), eps)
    weight_raw = np.maximum(1.0 - alpha * p_prime, eps)

    # f_w = f'_w / mean(f'_w)  --> mean weight becomes 1
    w = weight_raw / np.mean(weight_raw)

    return w.astype(np.float32)
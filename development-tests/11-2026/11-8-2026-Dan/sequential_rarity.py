"""Expert 1 rarity search for the sequential summed-expert experiments."""

from dataclasses import dataclass

import numpy as np


@dataclass
class CandidateFit:
    epochs: int
    fold_predictions: list
    validation_predictions: np.ndarray


def rarity_grid(targets, mode, candidates, rare_threshold):
    """Fit rarity on this training split's original targets, once for all candidates."""
    values = np.asarray(targets).reshape(-1)
    if mode == "residual_class":
        rare = values >= rare_threshold
        raw_weights = [np.where(rare, weight, 1.0) for weight in candidates]
    elif mode == "residual_reciprocal":
        from imbal.regression import fit_kde, get_sample_densities

        densities = np.ones(len(values))
        if np.ptp(values) > 0:
            bandwidth = fit_kde(values)
            densities = np.asarray(get_sample_densities(values, bandwidth), dtype="float64").ravel()
        if not np.all(np.isfinite(densities) & (densities > 0)):
            raise ValueError("Reciprocal importance requires finite positive target densities.")
        raw_weights = [densities ** -alpha for alpha in candidates]
    else:
        raise ValueError(f"Unknown rarity weighting mode: {mode}")

    return {value: (weights / weights.mean()).astype("float32")
            for value, weights in zip(candidates, raw_weights)}


def tune_first_expert(targets, folds, settings, rare_threshold, fit_candidate, regression_metrics):
    """Keep the winning fold predictions for the next expert and for its router."""
    grids = [rarity_grid(targets[fold.training_indices], settings.expert_mode,
                         settings.rarity_candidates, rare_threshold) for fold in folds]
    trials = []
    best_score = np.inf
    for value in settings.rarity_candidates:
        for fold, grid in zip(folds, grids):
            fold.rarity = grid[value]
        # fit_candidate resets the same fold seeds for every trial.
        result = fit_candidate(1)
        score = regression_metrics(targets, result.validation_predictions)["aore"]
        if score is None or not np.isfinite(score):
            raise ValueError("Expert 1 selection requires finite validation AORE and rare rows.")
        trials.append({"value": value, "validation_aore": score})
        print(f"Expert 1 {settings.expert_mode}={value}: validation AORE {score:.6f}")
        if score < best_score:
            best_score, best_value, best_result = score, value, result

    for fold, grid in zip(folds, grids):
        fold.rarity = grid[best_value]
    return best_result, {"rarity_method": settings.expert_mode, "selected_value": best_value,
                         "candidates": trials}

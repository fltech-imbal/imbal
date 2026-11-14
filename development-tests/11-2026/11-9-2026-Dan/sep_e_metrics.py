"""Regression metrics for the held-out folds and final evaluation splits."""

import numpy as np


RARE_THRESHOLD = np.log(10)


def regression_metrics(targets, predictions):
    targets = targets.ravel()
    errors = predictions.ravel().astype("float64") - targets
    absolute_errors = np.abs(errors)
    rare = targets >= RARE_THRESHOLD

    overall_mae = float(absolute_errors.mean())
    rare_mae = float(absolute_errors[rare].mean()) if rare.any() else None
    return {
        "aore": (overall_mae + rare_mae) / 2 if rare_mae is not None else None,
        "overall_mae": overall_mae,
        "common_mae": float(absolute_errors[~rare].mean()) if (~rare).any() else None,
        "rare_mae": rare_mae,
        "overall_mse": float(np.mean(np.square(errors))),
    }


def compare_second_expert(targets, first_predictions, prefix_predictions, residual=False):
    """Compare the same out-of-fold rows, including a target-informed routing bound."""

    targets = targets.ravel()
    first_predictions = first_predictions.ravel()
    prefix_predictions = prefix_predictions.ravel()
    gain = np.abs(targets - first_predictions) - np.abs(targets - prefix_predictions)
    oracle_predictions = np.where(gain > 0, prefix_predictions, first_predictions)

    rare_gain = gain[targets >= RARE_THRESHOLD]
    positive_gain = rare_gain[rare_gain > 0]
    return {
        "expert_1": regression_metrics(targets, first_predictions),
        ("two_expert_sum" if residual else "two_expert_mean"): regression_metrics(targets, prefix_predictions),
        "oracle": regression_metrics(targets, oracle_predictions),
        "rare_rows": int(rare_gain.size),
        "rare_improved": int(positive_gain.size),
        "rare_worsened": int((rare_gain < 0).sum()),
        "rare_tied": int((rare_gain == 0).sum()),
        "median_rare_error_reduction": float(np.median(rare_gain)) if rare_gain.size else None,
        "largest_share_of_rare_gains": (
            float(positive_gain.max() / positive_gain.sum()) if positive_gain.size else None
        ),
    }


def format_value(value):
    return "N/A" if value is None else f"{value:.6f}"


def average_metrics(metric_sets):
    # All seeds evaluate the same rows, so a missing subgroup is missing in every run.
    return {
        key: float(np.mean([metrics[key] for metrics in metric_sets])) if value is not None else None
        for key, value in metric_sets[0].items()
    }


def print_prefix_metrics(title, comparison):
    print(f"\n{title}")
    print(f"{'Predictor':<25} {'Overall MSE':>12} {'Common MAE':>12} {'Rare MAE':>12} {'AORE':>12}")
    metric_order = ("overall_mse", "common_mae", "rare_mae", "aore")
    prefix = "sum" if "two_expert_sum" in comparison else "mean"
    for name, key in (("Expert 1", "expert_1"), (f"Experts 1-2 {prefix}", f"two_expert_{prefix}"),
                      ("Best per row (oracle)", "oracle")):
        metrics = comparison[key]
        values = [format_value(metrics[metric]) for metric in metric_order]
        print(f"{name:<25}" + "".join(f" {value:>12}" for value in values))


def print_validation_comparison(comparison):
    print_prefix_metrics("Held-out validation before the Expert 2 decision", comparison)
    print(f"Rare rows: {comparison['rare_improved']} improved, {comparison['rare_worsened']} worsened, "
          f"{comparison['rare_tied']} tied when adding Expert 2.")
    print(f"Median rare absolute-error reduction: {format_value(comparison['median_rare_error_reduction'])}")
    share = comparison["largest_share_of_rare_gains"]
    if share is not None:
        print(f"Largest single-row share of total positive rare error reductions: {share:.1%}")

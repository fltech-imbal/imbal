import numpy as np


def _safe_mean(x):
    """Return mean or np.nan if empty."""
    return np.mean(x) if x.size > 0 else np.nan


def _pcc(a, b):
    """
    Safe Pearson correlation.
    Returns np.nan if not computable.
    """
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)

    if a.size < 2:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return np.corrcoef(a, b)[0, 1]


def compute_regression_metrics(y_true, y_pred, threshold):
    """
    Computes:
        - overall_mae
        - rare_mae
        - common_mae
        - final_mae = (rare_mae + overall_mae) / 2
        - overall_pcc
        - rare_pcc
        - final_pcc = (rare_pcc + overall_pcc) / 2
        - false_positives
        - false_negatives

    Rare defined as y_true >= threshold

    Returns tuple in fixed order:
        (
            overall_mae,
            rare_mae,
            common_mae,
            final_mae,
            overall_pcc,
            rare_pcc,
            final_pcc,
            fp,
            fn
        )
    """

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    # Masks
    rare_mask = y_true >= threshold
    common_mask = ~rare_mask

    # --- MAE ---
    abs_err = np.abs(y_true - y_pred)

    overall_mae = _safe_mean(abs_err)
    rare_mae = _safe_mean(abs_err[rare_mask])
    common_mae = _safe_mean(abs_err[common_mask])
    final_mae = 0.5 * (rare_mae + overall_mae)

    # --- PCC ---
    overall_pcc = _pcc(y_true, y_pred)
    rare_pcc = _pcc(y_true[rare_mask], y_pred[rare_mask])
    final_pcc = 0.5 * (rare_pcc + overall_pcc)

    # --- FP / FN ---
    pred_pos = y_pred >= threshold
    true_pos = y_true >= threshold

    fp = int(np.sum(pred_pos & ~true_pos))
    fn = int(np.sum(~pred_pos & true_pos))

    return (
        overall_mae,
        rare_mae,
        common_mae,
        final_mae,
        overall_pcc,
        rare_pcc,
        final_pcc,
        fp,
        fn,
    )
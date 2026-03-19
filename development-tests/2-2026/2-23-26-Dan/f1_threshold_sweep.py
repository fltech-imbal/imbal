import numpy as np

def f1_threshold_sweep(y_prob, y_test, step=0.1, verbose=1):
    """
    Sweep thresholds from 0.1 to 0.9 using the given step size,
    compute confusion matrix + F1 at each threshold,
    and return the best threshold.

    Parameters:
        y_prob : array-like
            Predicted probabilities (NOT binary predictions).
        y_test : array-like
            True binary labels (0/1).
        step : float
            Threshold increment (e.g., 0.05).

    Returns:
        best_threshold (float)
        best_f1 (float)
        best_counts (dict)
        results (list of dicts)
    """

    # Ensure correct shape
    y_prob = np.asarray(y_prob).reshape(-1)
    y_true = np.asarray(y_test).astype(int).reshape(-1)

    eps = 1e-12
    thresholds = np.arange(0 + eps, 1 - eps, step)

    results = []
    best = {"threshold": None, "f1": -1.0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        TP = int(np.sum((y_true == 1) & (y_pred == 1)))
        TN = int(np.sum((y_true == 0) & (y_pred == 0)))
        FP = int(np.sum((y_true == 0) & (y_pred == 1)))
        FN = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        row = {
            "threshold": float(t),
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        results.append(row)

        # Keep lower threshold if tied (optional behavior)
        if (f1 > best["f1"]) or (np.isclose(f1, best["f1"]) and (best["threshold"] is None or t < best["threshold"])):
            best = {"threshold": float(t), "f1": float(f1), "TP": TP, "TN": TN, "FP": FP, "FN": FN}

    if verbose == 1:
        print("\n=== Threshold Sweep (F1) ===")
        print(f"Best threshold: {best['threshold']:.4f}")
        print(f"Best F1:        {best['f1']:.4f}")
        print("=== Confusion Matrix @ Best Threshold ===")
        print(f"TP: {best['TP']}")
        print(f"FP: {best['FP']}")
        print(f"TN: {best['TN']}")
        print(f"FN: {best['FN']}")

    best_counts = {"TP": best["TP"], "TN": best["TN"], "FP": best["FP"], "FN": best["FN"]}
    return best["threshold"], best["f1"], best_counts, results

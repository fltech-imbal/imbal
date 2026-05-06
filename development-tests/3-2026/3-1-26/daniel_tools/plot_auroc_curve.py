# Put this near your imports (top of file)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib.collections import LineCollection


def plot_auroc_curve(
    model,
    x,
    y_true,
    *,
    title="AUROC Curve",
    color_by_threshold=False,
    drop_intermediate=False,
    save_path=None,
    show=True,
):
    """
    Plots an ROC curve and prints AUROC.

    Args:
        model: Keras model with predict().
        x: Features to score (e.g., x_test).
        y_true: True labels (shape (N,) or (N,1)), values in {0,1} or bool.
        title: Plot title.
        color_by_threshold: If True, color curve by decision threshold.
        drop_intermediate: Passed to sklearn.metrics.roc_curve.
        save_path: If provided, saves figure to this path.
        show: If True, plt.show().

    Returns:
        (fpr, tpr, thresholds, roc_auc)
    """
    # Ensure shapes/types sklearn expects
    y_true_1d = np.asarray(y_true).reshape(-1).astype(int)

    # Probability scores in [0,1]
    y_score = model.predict(x, verbose=0).reshape(-1)

    fpr, tpr, thresholds = roc_curve(
        y_true_1d, y_score, drop_intermediate=drop_intermediate
    )
    roc_auc = auc(fpr, tpr)
    print(f"sklearn AUROC: {roc_auc:.6f}")

    if not color_by_threshold:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        else:
            plt.close()

        return fpr, tpr, thresholds, roc_auc

    # ---- Color-by-threshold version ----
    # Build segments for a colored line
    points = np.array([fpr, tpr]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    thr = thresholds.copy()
    thr[np.isinf(thr)] = 1.0  # make inf plot-friendly (optional)

    lc = LineCollection(
        segments,
        cmap="viridis",
        norm=plt.Normalize(vmin=0.0, vmax=1.0),
    )
    # segments are one shorter than points
    lc.set_array(thr[:-1])
    lc.set_linewidth(2)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.add_collection(lc)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title} (AUC = {roc_auc:.4f})")
    ax.grid(True)

    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Decision Threshold")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

    return fpr, tpr, thresholds, roc_auc

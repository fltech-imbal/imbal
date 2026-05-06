"""
sep_regression_plot.py

Utilities to evaluate and plot regression results for ln(peak intensity).

Requirements:
- numpy
- matplotlib

Usage example:

from sep_regression_plot import eval_and_plot_regression

y_true = test_data["ln_peak_intensity"].to_numpy()
y_pred = model.predict(x_test).reshape(-1)

metrics = eval_and_plot_regression(
    y_true=y_true,
    y_pred=y_pred,
    threshold=np.log(10.0),
    out_png="pred_vs_actual.png",
    title="Predicted vs Actual ln(peak intensity)",
)
print(metrics)
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def _to_1d_float(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D (or shape (n,1)). Got shape {arr.shape}.")
    return arr.astype(np.float64)


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    out_png: Optional[str] = None,
    title: str = "Predicted vs Actual ln(peak intensity)",
    alpha: float = 0.7,
    s: float = 14.0,
    show: bool = True,
) -> None:
    """
    Scatter plot:
      x-axis: actual
      y-axis: predicted
      red: SEP events (actual >= threshold)
      blue: non-SEP events (actual < threshold)
      dotted lines: y=x, x=threshold, y=threshold
    """
    y_true = _to_1d_float(y_true, "y_true")
    y_pred = _to_1d_float(y_pred, "y_pred")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true and y_pred must have same length. Got {len(y_true)} vs {len(y_pred)}.")

    sep_mask = y_true >= threshold
    non_mask = ~sep_mask

    # Axis limits that include both true and predicted, plus the threshold
    all_vals = np.concatenate([y_true, y_pred, np.array([threshold], dtype=np.float64)])
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))

    # Small padding so points and lines aren't on the border
    pad = 0.03 * (vmax - vmin) if vmax > vmin else 1.0
    xmin, xmax = vmin - pad, vmax + pad
    ymin, ymax = xmin, xmax  # square view makes y=x visually meaningful

    plt.figure(figsize=(7.5, 7.5))

    # Points: non-SEP first, then SEP on top
    plt.scatter(
        y_true[non_mask],
        y_pred[non_mask],
        c="blue",
        label=f"non-SEP (actual < ln(10))  n={int(np.sum(non_mask))}",
        alpha=alpha,
        s=s,
        edgecolors="none",
    )
    plt.scatter(
        y_true[sep_mask],
        y_pred[sep_mask],
        c="red",
        label=f"SEP (actual ≥ ln(10))  n={int(np.sum(sep_mask))}",
        alpha=alpha,
        s=s,
        edgecolors="none",
    )

    # Dotted reference lines
    # diagonal y=x
    plt.plot([xmin, xmax], [ymin, ymax], linestyle=":", linewidth=1.5, label="y = x")

    # x=threshold vertical
    plt.axvline(threshold, linestyle=":", linewidth=1.5)
    # y=threshold horizontal
    plt.axhline(threshold, linestyle=":", linewidth=1.5)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.xlabel("Actual ln(peak intensity)")
    plt.ylabel("Predicted ln(peak intensity)")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.metrics import roc_curve

def plot_roc(
    labels,
    predictions,
    save_figure=None,
):
    labels = labels.reshape(-1, 1)
    predictions = predictions.reshape(-1, 1)
    fpr, tpr, thresholds = roc_curve(labels, predictions, drop_intermediate=False)
    plt.figure(figsize=(7, 6))
    points = np.array([fpr, tpr]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm_thresholds = thresholds

    lc = LineCollection(
        segments,
        cmap='viridis',
        norm=plt.Normalize(vmin=0, vmax=1)
    )
    lc.set_array(norm_thresholds)
    lc.set_linewidth(2)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.add_collection(lc)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Decision Threshold")

    if save_figure is not None:
        plt.savefig(save_figure)
    plt.show()
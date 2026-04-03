import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(
    labels,
    predictions,
    save_figure=None
):
    cm = confusion_matrix(labels.reshape(-1), np.round(predictions).reshape(-1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion matrix for test data')

    # Save plot if path is specified
    if save_figure is not None:
        plt.savefig(save_figure)
    plt.show()
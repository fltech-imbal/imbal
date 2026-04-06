import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(
    labels,
    predictions,
    save_figure=None
):
    """
    Plots and displays a confusion matrix for the provided labels and predictions.

    Args:
        labels: A NumPy array of binary labels (0/1 or true/false)
        predictions: A NumPy array of predictions corresponding to the provided labels
        save_figure: Optional, default :code:`None`. If set to a string, will save the
            resultant plot to the specified path.

    Example:

    .. code::

        from imbal.classification import plot_confusion_matrix
        import numpy as np

        labels = np.array([0, 0, 1, 1])
        predictions = np.array([0, 1, 0, 1])

        # Plots a confusion matrix with one true positive, one false
        # positive, one false negative, and one true negative
        plot_confusion_matrix(labels, predictions)

    """

    cm = confusion_matrix(labels.reshape(-1), np.round(predictions).reshape(-1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion matrix for test data')

    # Save plot if path is specified
    if save_figure is not None:
        plt.savefig(save_figure)
    plt.show()
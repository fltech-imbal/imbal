import matplotlib.pyplot as plt
import numpy as np

def plot_true_vs_predictions(
    labels,
    predictions,
    x_axis_label=None,
    y_axis_label=None,
    title=None,
    color=None,
    marker=None,
    size=None,
    save_figure=None
):
    """
    Plots and displays a comparison of the true labels and the predicted labels
    for regression data. Points plotted are (true, prediction) for each label/prediction
    pair, such that a perfect prediction would fall precisely on the line :math:`y=x`.
    Prediction-label pairs will be plotted in the same order as supplied to the function.

    Args:
        labels: A NumPy array of binary labels (0/1 or true/false)
        predictions: A NumPy array of predictions corresponding to the provided labels
        x_axis_label: Optional, default :code:`None`. A string to label the x-axis.
        y_axis_label: Optional, default :code:`None`. A string to label the y-axis.
        title: Optional, default :code:`None`. A string to title the generated plot.
        color: Optional, default :code:`None`. A color to apply to the points within the
            generated plot. See `matplotlib.pyplot.scatter <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html>`_.
        marker: Optional, default :code:`None`. A marker to use when plotting points within the
            generated plot. See `matplotlib.pyplot.scatter <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html>`_.
        size: Optional, default :code:`None`. A size to plot each point within the
            generated plot. See `matplotlib.pyplot.scatter <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html>`_.
        save_figure: Optional, default :code:`None`. If set to a string, will save the
            resultant plot to the specified path.

    Example:

    .. code::

        from imbal.regression import plot_true_vs_predictions
        import numpy as np

        labels = np.array([0, 1, 2, 3, 4, 5])
        predictions = np.array([0, 1, 1, 3.5, 4.75])

        # Plots the comparison and saves it to "true_vs_predictions.png"
        plot_true_vs_predictions(
            labels,
            predictions,
            save_figure="true_vs_predictions.png"
        )

    """
    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)

    data_min = np.min([labels.min(), predictions.min()]) - 1
    data_max = np.max([labels.max(), predictions.max()]) + 1

    # Create comparison plot
    plt.figure(figsize=(7, 6))
    plt.plot([data_min, data_max], [data_min, data_max], linestyle="--", linewidth=1, color='black', label="Perfect Prediction")
    plt.scatter(labels, predictions, color="#00FF0044" if color is None else color, s=size, marker=marker)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.xlabel("True Label" if x_axis_label is None else x_axis_label)
    plt.ylabel("Predicted Label" if y_axis_label is None else y_axis_label)
    plt.xlim(data_min, data_max)
    plt.ylim(data_min, data_max)
    if title is not None:
        plt.title(title)
    if save_figure is not None:
        plt.savefig(save_figure)
    # Save plot if path is specified
    if save_figure is not None:
        plt.savefig(save_figure)
    plt.show()
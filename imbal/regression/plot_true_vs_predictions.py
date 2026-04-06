import matplotlib.pyplot as plt

def plot_true_vs_predictions(
    labels,
    predictions,
    rare_threshold=-4,
    low_bound=-9.5,
    high_bound=-2,
    save_figure=None
):
    """
    Plots and displays a comparison of the true labels and the predicted labels
    for regression data. Points plotted are (true, prediction) for each label/prediction
    pair, such that a perfect prediction would fall precisely on the line :math:`y=x`.

    Args:
        labels: A NumPy array of binary labels (0/1 or true/false)
        predictions: A NumPy array of predictions corresponding to the provided labels
        rare_threshold: The threshold for what is considered a "rare instance"
        low_bound: The lower bound of the x/y range displayed in the plot
        high_bound: The upper bound of the x/y range displayed in the plot
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
            rare_threshold=4,
            low_bound=-0.5,
            high_bound=5.5,
            save_figure="true_vs_predictions.png"
        )

    """
    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)

    # Mask rare and frequent data
    rare_mask = labels > rare_threshold
    frequent_mask = ~rare_mask
    frequent_labels = labels[frequent_mask]
    frequent_predictions = predictions[frequent_mask]
    rare_labels = labels[rare_mask]
    rare_predictions = predictions[rare_mask]

    # Create comparison plot
    plt.figure(figsize=(7, 6))
    plt.plot([low_bound, high_bound], [low_bound, high_bound], linestyle="--", linewidth=1, color='black', label="Perfect Prediction")
    light_gray = '#BBBBBB'
    plt.plot([rare_threshold, rare_threshold], [low_bound, high_bound], linestyle="--", linewidth=1, color=light_gray)
    plt.plot([low_bound, high_bound], [rare_threshold, rare_threshold], linestyle="--", linewidth=1, color=light_gray)
    plt.scatter(frequent_labels, frequent_predictions, color="#00FF00", alpha=0.3)
    plt.scatter(rare_labels, rare_predictions, color="#FF0000", alpha=0.2)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.xlim(low_bound, high_bound)
    plt.ylim(low_bound, high_bound)
    # Save plot if path is specified
    if save_figure is not None:
        plt.savefig(save_figure)
    plt.show()
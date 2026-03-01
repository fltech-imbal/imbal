import matplotlib.pyplot as plt

def plot_true_vs_predictions(
    labels,
    predictions,
    save_figure=None
):
    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)

    plt.figure(figsize=(7, 6))
    plt.plot([-2.5, 2.5], [-2.5, 2.5], linestyle="--", linewidth=1, color='black', label="Perfect Prediction")
    plt.scatter(labels, predictions.reshape(-1), color="#FF0000", alpha=0.2)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    if save_figure is not None:
        plt.savefig(save_figure)
    plt.show()
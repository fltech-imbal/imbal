import matplotlib.pyplot as plt

def plot_true_vs_predictions(
    labels,
    predictions,
    save_figure=None
):
    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)

    rare_mask = (labels < -0.5) | (labels > 0.5)
    frequent_mask = ~rare_mask

    frequent_labels = labels[frequent_mask]
    frequent_predictions = predictions[frequent_mask]
    rare_labels = labels[rare_mask]
    rare_predictions = predictions[rare_mask]

    plt.figure(figsize=(7, 6))
    plt.plot([-2.5, 2.5], [-2.5, 2.5], linestyle="--", linewidth=1, color='black', label="Perfect Prediction")
    light_gray = '#BBBBBB'
    plt.plot([-0.5, -0.5], [-2.5, 2.5], linestyle="--", linewidth=1, color=light_gray)
    plt.plot([0.5, 0.5], [-2.5, 2.5], linestyle="--", linewidth=1, color=light_gray)
    plt.plot([-2.5, 2.5], [-0.5, -0.5], linestyle="--", linewidth=1, color=light_gray)
    plt.plot([-2.5, 2.5], [0.5, 0.5], linestyle="--", linewidth=1, color=light_gray)
    plt.scatter(frequent_labels, frequent_predictions, color="#00FF00", alpha=0.3)
    plt.scatter(rare_labels, rare_predictions, color="#FF0000", alpha=0.2)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    if save_figure is not None:
        plt.savefig(save_figure)
    plt.show()
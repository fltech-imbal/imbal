import os
import json
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt

import imbal


# ----------------------------
# Data
# ----------------------------
target_column = "ln_peak_intensity"

test_data = pd.read_csv("../../../../tutorials/data/SEP-C/sep_10mev_testing_classification.csv")

y_test = test_data[target_column].values.reshape(-1, 1).astype("float32")
x_test = test_data.drop(columns=[target_column]).values.astype(np.float32)


# ----------------------------
# Model paths
# ----------------------------
model_paths = [
    "saved_models/regular-fit-model.keras",
    "saved_models/regular-fit-model-val.keras",
    "saved_models/regular-fit-model-val-ae.keras",
    "saved_models/regular-fit-model-val-ae-rep-3.keras",
    "saved_models/balanced-fit-model.keras",
    "saved_models/pseudo-label-generator-model.keras",
    "saved_models/balanced-fit-model-val-ae.keras",
    "saved_models/balanced-fit-model-val-ae-rep-3.keras",
    "saved_models/decoupled-fit-model.keras",
    "saved_models/decoupled-fit-model-val.keras",
    "saved_models/decoupled-fit-model-val-ae.keras",
    "saved_models/decoupled-fit-model-val-ae-rep-3.keras",
]

best_param_paths = [
    "saved_models/best_params_regular_fit.json",
    "saved_models/best_params_regular_fit-val.json",
    "saved_models/best_params_regular_fit-val-ae.json",
    "saved_models/best_params_regular_fit-val-ae-rep-3.json",
    "saved_models/best_params_balanced_fit.json",
    "saved_models/best_params_balanced_fit-val.json",
    "saved_models/best_params_balanced_fit-val-ae.json",
    "saved_models/best_params_balanced_fit-val-ae-rep-3.json",
    "saved_models/best_params_decoupled_fit.json",
    "saved_models/best_params_decoupled_fit-val.json",
    "saved_models/best_params_decoupled_fit-val-ae.json",
    "saved_models/best_params_decoupled_fit-val-ae-rep-3.json",
]


def load_best_params(params_path):
    if not os.path.exists(params_path):
        return None

    with open(params_path, "r") as f:
        return json.load(f)


def metric_result_to_float(metric_result):
    metric_result = metric_result.numpy()

    if np.ndim(metric_result) == 0:
        return float(metric_result)

    return float(metric_result.flatten()[0])


def find_best_f1_threshold(y_true, y_pred_probs):
    best_threshold = None
    best_f1_value = None

    for threshold in np.arange(0.1, 1.0, 0.1):
        f1 = keras.metrics.F1Score(threshold=threshold)
        f1.update_state(y_true, y_pred_probs)

        f1_value = metric_result_to_float(f1.result())

        if best_f1_value is None or f1_value > best_f1_value:
            best_f1_value = f1_value
            best_threshold = threshold

    return best_threshold, best_f1_value


def make_function_figure_image(plot_function, error_message):
    old_show = plt.show
    plt.show = lambda *args, **kwargs: None

    existing_figures = set(plt.get_fignums())

    try:
        plot_function()
    finally:
        plt.show = old_show

    new_figures = [
        figure_number
        for figure_number in plt.get_fignums()
        if figure_number not in existing_figures
    ]

    if len(new_figures) == 0:
        raise RuntimeError(error_message)

    generated_figure = plt.figure(new_figures[-1])
    generated_figure.canvas.draw()

    image = np.asarray(generated_figure.canvas.buffer_rgba()).copy()

    plt.close(generated_figure)

    return image


def make_confusion_matrix_image(y_true, y_pred):
    return make_function_figure_image(
        plot_function=lambda: imbal.classification.plot_confusion_matrix(
            y_true,
            y_pred,
            square_font_size=16
        ),
        error_message="imbal.classification.plot_confusion_matrix did not create a figure.",
    )


def make_tsne_visualization_image(model, x_data, y_true):
    return make_function_figure_image(
        plot_function=lambda: imbal.classification.tsne_visualization(
            model,
            x_data,
            y_true
        ),
        error_message="imbal.classification.tsne_visualization did not create a figure.",
    )


# ----------------------------
# Plot settings
# ----------------------------
num_rows = 3
num_cols = 4

confusion_fig, confusion_axes = plt.subplots(
    num_rows,
    num_cols,
    figsize=(20, 15)
)
confusion_axes = confusion_axes.flatten()

tsne_fig, tsne_axes = plt.subplots(
    num_rows,
    num_cols,
    figsize=(20, 15)
)
tsne_axes = tsne_axes.flatten()


# ----------------------------
# Load, predict, evaluate, plot
# ----------------------------
for i, model_path in enumerate(model_paths):
    confusion_ax = confusion_axes[i]
    tsne_ax = tsne_axes[i]

    model = keras.models.load_model(
        model_path,
        custom_objects={
            "Model": imbal.classification.Model,
            "HeidkeSkillScore": imbal.metrics.HeidkeSkillScore,
        },
    )

    y_pred_probs = model.predict(x_test).reshape(-1, 1)

    best_params = load_best_params(best_param_paths[i])

    if best_params is not None and best_params["best_decision_threshold"] != -1:
        best_threshold = float(best_params["best_decision_threshold"])
    else:
        print(f"No best_decision_threshold found for {model_path}")
        best_threshold, _ = find_best_f1_threshold(
            y_test,
            y_pred_probs
        )

    y_pred = (
        y_pred_probs > best_threshold
    ).astype(np.float32)

    f1 = keras.metrics.F1Score(
        threshold=best_threshold
    )
    f1.update_state(
        y_test,
        y_pred_probs
    )
    f1_value = metric_result_to_float(
        f1.result()
    )

    hss = imbal.metrics.HeidkeSkillScore(
        threshold=best_threshold
    )
    hss.update_state(
        y_test,
        y_pred_probs
    )
    hss_value = metric_result_to_float(
        hss.result()
    )

    class_weights = None

    if best_params is not None and best_params["best_class_weights"] != -1:
        class_weights = best_params["best_class_weights"]

    confusion_matrix_image = make_confusion_matrix_image(
        y_test,
        y_pred
    )

    tsne_visualization_image = make_tsne_visualization_image(
        model,
        x_test,
        y_test.reshape(-1)
    )

    model_name = os.path.basename(model_path)

    title = (
        f"{model_name}\n"
        f"Threshold: {best_threshold:.2f}"
    )

    if class_weights is not None:
        title += f" | Class weights: {class_weights}"

    title += (
        f"\nF1: {f1_value:.4f} | "
        f"HSS: {hss_value:.4f}"
    )

    confusion_ax.imshow(confusion_matrix_image)
    confusion_ax.axis("off")
    confusion_ax.set_title(
        title,
        fontsize=9
    )

    tsne_ax.imshow(tsne_visualization_image)
    tsne_ax.axis("off")
    tsne_ax.set_title(
        title,
        fontsize=9
    )


confusion_fig.suptitle(
    "Confusion Matrices",
    fontsize=16
)
confusion_fig.tight_layout()

tsne_fig.suptitle(
    "t-SNE Visualizations",
    fontsize=16
)
tsne_fig.tight_layout()

plt.show()
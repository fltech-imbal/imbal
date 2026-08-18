import os
import json
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt

import imbal
from aore_metric import AORE


# ----------------------------
# Data
# ----------------------------
target_column = "ln_peak_intensity"

test_data = pd.read_csv("../../../../tutorials/data/SEP-C/sep_10mev_testing.csv")

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
    "saved_models/balanced-fit-model-val.keras",
    "saved_models/balanced-fit-model-val-ae.keras",
    "saved_models/balanced-fit-model-val-ae-rep-3.keras",
    "saved_models/decoupled-fit-model.keras",
    "saved_models/decoupled-fit-model-val.keras",
    "saved_models/decoupled-fit-model-val-ae.keras",
    "saved_models/decoupled-fit-model-val-ae-rep-3.keras",
]

best_param_paths = [
    "saved_models/best_params_regular_fit_regression.json",
    "saved_models/best_params_regular_fit_regression-val.json",
    "saved_models/best_params_regular_fit_regression-val-ae.json",
    "saved_models/best_params_regular_fit_regression-val-ae-rep-3.json",
    "saved_models/best_params_balanced_fit_regression.json",
    "saved_models/best_params_balanced-fit-regression-val.json",
    "saved_models/best_params_balanced_fit_regression-val-ae.json",
    "saved_models/best_params_balanced_fit_regression-val-ae-rep-3.json",
    "saved_models/best_params_decoupled_fit_regression.json",
    "saved_models/best_params_decoupled_fit_regression-val.json",
    "saved_models/best_params_decoupled_fit_regression-val-ae.json",
    "saved_models/best_params_decoupled_fit_regression-val-ae-rep-3.json",
]


def load_best_params(params_path):
    if not os.path.exists(params_path):
        return None

    with open(params_path, "r") as f:
        return json.load(f)


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


def make_regression_plot_image(y_true, y_pred):
    return make_function_figure_image(
        plot_function=lambda: imbal.regression.plot_true_vs_predictions(
            y_true,
            y_pred
        ),
        error_message="imbal.regression.plot_true_vs_predictions did not create a figure.",
    )


def make_tsne_visualization_image(model, x_data, y_true):
    return make_function_figure_image(
        plot_function=lambda: imbal.regression.tsne_visualization(
            model,
            x_data,
            y_true
        ),
        error_message="imbal.regression.tsne_visualization did not create a figure.",
    )


def compute_regression_metrics(y_true, y_pred, threshold=np.log(10)):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    absolute_errors = np.abs(y_true - y_pred)

    common_mask = y_true < threshold
    rare_mask = y_true >= threshold

    common_mae = np.mean(absolute_errors[common_mask])
    rare_mae = np.mean(absolute_errors[rare_mask])

    overall_mae = np.mean(absolute_errors)
    aore = (overall_mae + rare_mae) / 2.0

    return common_mae, rare_mae, aore


def get_best_alpha_text(best_params):
    if best_params is None:
        return ""

    best_alpha_index = best_params.get("best_alpha_index", -1)
    best_alpha = best_params.get("best_alpha", -1)

    if best_alpha_index == -1 and best_alpha == -1:
        return ""

    return f"\nBest alpha index: {best_alpha_index} | Best alpha: {best_alpha}"


# ----------------------------
# Plot settings
# ----------------------------
num_rows = 3
num_cols = 4

regression_fig, regression_axes = plt.subplots(
    num_rows,
    num_cols,
    figsize=(20, 15)
)
regression_axes = regression_axes.flatten()

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
    regression_ax = regression_axes[i]
    tsne_ax = tsne_axes[i]

    model = keras.models.load_model(
        model_path,
        custom_objects={
            "Model": imbal.regression.Model,
            "AORE": AORE,
        },
    )

    predictions = model.predict(x_test).reshape(-1, 1)

    common_mae, rare_mae, aore = compute_regression_metrics(
        y_test,
        predictions,
        threshold=np.log(10),
    )

    best_params = load_best_params(best_param_paths[i])

    regression_plot_image = make_regression_plot_image(
        y_test,
        predictions
    )

    tsne_visualization_image = make_tsne_visualization_image(
        model,
        x_test,
        y_test.reshape(-1)
    )

    model_name = os.path.basename(model_path)

    title = (
        f"{model_name}"
        f"{get_best_alpha_text(best_params)}"
        f"\nCommon MAE: {common_mae:.4f} | "
        f"Rare MAE: {rare_mae:.4f} | "
        f"AORE: {aore:.4f}"
    )

    regression_ax.imshow(regression_plot_image)
    regression_ax.axis("off")
    regression_ax.set_title(
        title,
        fontsize=9
    )

    tsne_ax.imshow(tsne_visualization_image)
    tsne_ax.axis("off")
    tsne_ax.set_title(
        title,
        fontsize=9
    )


regression_fig.suptitle(
    "True vs. Predicted Regression Plots",
    fontsize=16
)
regression_fig.tight_layout()

tsne_fig.suptitle(
    "t-SNE Visualizations",
    fontsize=16
)
tsne_fig.tight_layout()

plt.show()
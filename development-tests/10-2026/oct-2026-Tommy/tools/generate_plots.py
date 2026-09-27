import tensorflow as tf
import numpy as np
from .plot_representation_space import plot_representation_space_3d, plot_representation_space
import imbal
import os

def generate_plots(
    x,
    y,
    model,
    common_mask,
    path_prefix,
    three_d=False,
    vmin=None,
    vmax=None,
    plot_names='temp'
):
    predictions, representations, pre_representations = model.predict(x)
    predictions = predictions.reshape(-1)

    if isinstance(y, tf.Tensor):
        y = y.numpy()

    y = y.reshape(-1)
    common_mask = common_mask.reshape(-1)

    common_predictions = predictions[common_mask]
    rare_predictions = predictions[~common_mask]
    common_labels = y[common_mask]
    rare_labels = y[~common_mask]

    mae = np.mean(np.abs(predictions - y))
    common_mae = np.mean(np.abs(common_predictions - common_labels))
    rare_mae = np.mean(np.abs(rare_predictions - rare_labels))

    if not os.path.exists(f'{path_prefix}/pre_representation'):
        os.makedirs(f'{path_prefix}/pre_representation')
    if not os.path.exists(f'{path_prefix}/representation'):
        os.makedirs(f'{path_prefix}/representation')
    if not os.path.exists(f'{path_prefix}/true_vs_predicted'):
        os.makedirs(f'{path_prefix}/true_vs_predicted')

    imbal.regression.tsne_visualization(
        model,
        x,
        y,
        -3,
        save_figure=None if plot_names is None else f'{path_prefix}/pre_representation/{plot_names}.png'
    )

    if three_d:
        plot_representation_space_3d(
            representations,
            y,
            save_figure=None if plot_names is None else f'{path_prefix}/representation/{plot_names}.png',
            vmin=vmin,
            vmax=vmax
        )


    else:
        plot_representation_space(
            representations,
            y,
            save_figure=f'{path_prefix}/representation/{plot_names}.png',
            vmin=vmin,
            vmax=vmax
        )

    imbal.regression.plot_true_vs_predictions(
        y,
        predictions,
        title=f'Common MAE: {common_mae:.4f} ({np.count_nonzero(common_mask)}), Rare MAE: {rare_mae:.4f} ({np.count_nonzero(~common_mask)}), AORE: {(mae + rare_mae) / 2:.4f}',
        save_figure=None if plot_names is None else f'{path_prefix}/true_vs_predicted/{plot_names}.png'
    )

    return common_mae, rare_mae, (mae + rare_mae) / 2
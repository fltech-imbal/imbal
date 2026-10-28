"""Visualize the saved median-AORE SEP-C balanced-fit regression model (no training)."""
import json
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import imbal
from imbal.util.backend.explanation import shap_explain_tabular_sample

MODEL_PATH = 'saved_models/balanced-fit-model-val.keras'
PARAMS_PATH = 'saved_models/median_params_balanced_fit_regression-val.json'
RESULTS_PATH = 'saved_results/multi_seed_results_balanced_fit_regression-val.json'
TRAIN_PATH = '../../../../tutorials/data/SEP-C/sep_10mev_training.csv'
TEST_PATH = '../../../../tutorials/data/SEP-C/sep_10mev_testing.csv'
OUTPUT_DIR = 'saved_results/median_balanced_fit_regression_visualizations'
TARGET = 'ln_peak_intensity'
RARE_THRESHOLD = np.log(10)
RANDOM_STATE = 42
SHAP_BACKGROUND_SIZE = 100


# Same feature plots and fixed legend locations as the SEP-C classification visualizer.
FEATURE_PLOTS = [
    ('Linear Speed', 'CME_DONKI_speed_norm', 'feature_linear_speed.png', 'upper left'),
    ('Longitude', 'CME_DONKI_longitude_norm', 'feature_longitude.png', 'upper left'),
    ('Latitude', 'CME_DONKI_latitude_norm', 'feature_latitude.png', 'upper left'),
    ('Half Width', 'DONKI_half_width_norm', 'feature_half_width.png', 'upper left'),
    ('Acceleration', 'Accelaration_norm', 'feature_acceleration.png', 'upper right'),
]


def plot_feature_analysis(test, y_true, y_pred):
    """Save five individual feature plots using ln(10) as the event threshold."""
    if len(test) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError('Feature rows, true targets, and predictions must align.')
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError('Actual and predicted targets must be finite.')

    actual_rare = y_true >= RARE_THRESHOLD
    predicted_rare = y_pred >= RARE_THRESHOLD
    fp = ~actual_rare & predicted_rare
    fn = actual_rare & ~predicted_rare
    background = y_true <= 0.0
    elevated = (y_true > 0.0) & (y_true < RARE_THRESHOLD)

    print(f'Plot categories: Background={background.sum()}, Elevated={elevated.sum()}, '
          f'Rare={actual_rare.sum()}, FP={fp.sum()}, FN={fn.sum()}')

    for title, column, filename, legend_location in FEATURE_PLOTS:
        if column not in test.columns:
            raise ValueError(f'Missing feature {column} for {title}. '
                             f'Available columns: {list(test.columns)}')
        x = pd.to_numeric(test[column], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError(f'Nonfinite feature values in {column}.')

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        # Misclassified samples are triangles ONLY, not colored circles underneath.
        for mask, label, color in [
            (background & ~fp, 'Background', 'blue'),
            (elevated & ~fp, 'Elevated', 'green'),
            (actual_rare & ~fn, 'SEP', 'red'),
        ]:
            ax.scatter(x[mask], y_pred[mask], s=15, alpha=1.0,
                       c=color, marker='o', label=label, edgecolors='none', zorder=2)
        ax.scatter(x[fp], y_pred[fp], s=28, c='black', marker='^',
                   label='FP', edgecolors='none', zorder=4)
        ax.scatter(x[fn], y_pred[fn], s=28, c='black', marker='v',
                   label='FN', edgecolors='none', zorder=4)
        ax.axhline(RARE_THRESHOLD, color='gray', linestyle='--', linewidth=1.1)
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel('Predicted ln peak intensity')
        ax.legend(loc=legend_location, frameon=True, fontsize=9)
        fig.tight_layout()
        output = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(output, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {output}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PARAMS_PATH, encoding='utf-8') as f:
        params = json.load(f)
    with open(RESULTS_PATH, encoding='utf-8') as f:
        results = json.load(f)

    # The five-run averages are reported separately from the median model's analysis.
    runs = results['all_seed_results']
    if len(runs) != 5:
        raise ValueError(f'Expected five runs, found {len(runs)}.')
    print('Average results across five runs:')
    for metric in ('overall_mae', 'common_mae', 'rare_mae', 'aore'):
        print(f'  {metric}: {np.mean([run[metric] for run in runs]):.4f}')
    print(f"Median model seed: {params['median_seed']}")
    print(f"Median model AORE: {params['aore']:.4f}")
    print(f"Median model best alpha: {params['best_alpha']}")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    feature_names = train.drop(columns=[TARGET]).columns.tolist()
    x_train = train[feature_names].to_numpy(dtype=np.float32)
    x_test = test[feature_names].to_numpy(dtype=np.float32)
    y_test = test[TARGET].to_numpy(dtype=np.float32).ravel()

    model = keras.models.load_model(
        MODEL_PATH, compile=False,
        custom_objects={'Model': imbal.regression.Model},
    )
    y_pred = np.asarray(model.predict(x_test, verbose=0)).ravel()
    errors = np.abs(y_pred - y_test)

    plot_feature_analysis(test, y_test, y_pred)

    # Actual vs predicted using imbal's regression visualization.
    imbal.regression.plot_true_vs_predictions(
        y_test.reshape(-1, 1),
        y_pred.reshape(-1, 1),
    )
    plt.savefig(os.path.join(OUTPUT_DIR, 'actual_vs_predicted.png'), dpi=200, bbox_inches='tight')
    plt.close('all')

    # Single t-SNE for the saved median regression model.
    if len(x_test) >= 3:
        imbal.regression.tsne_visualization(
            model, x_test, y_test,
            save_figure=os.path.join(OUTPUT_DIR, 'tsne_model.png'),
            perplexity=250,
        )
        plt.close('all')

    # False negative: actual rare, predicted common.
    # False positive: actual common, predicted rare.
    # Select the largest absolute residual in each group (furthest from y=x).
    fn_indices = np.flatnonzero((y_test >= RARE_THRESHOLD) & (y_pred < RARE_THRESHOLD))
    fp_indices = np.flatnonzero((y_test < RARE_THRESHOLD) & (y_pred >= RARE_THRESHOLD))
    print(f'False negatives: {len(fn_indices)}; false positives: {len(fp_indices)}')

    rng = np.random.default_rng(RANDOM_STATE)
    background_indices = rng.choice(len(x_train), min(SHAP_BACKGROUND_SIZE, len(x_train)), replace=False)
    background = x_train[background_indices]

    for name, indices in [('worst_fn', fn_indices), ('worst_fp', fp_indices)]:
        if len(indices) == 0:
            print(f'No {name} examples; skipping SHAP.')
            continue
        selected = int(indices[np.argmax(errors[indices])])
        print(f'{name}: row={selected}, actual={y_test[selected]:.6f}, '
              f'predicted={y_pred[selected]:.6f}, absolute error={errors[selected]:.6f}')
        shap_explain_tabular_sample(
            sample=x_test[selected],
            model=model,
            training_data=background,
            feature_names=feature_names,
            actual_label=float(y_test[selected]),
            label_to_explain=None,
            class_names=None,
            plot_type='bar',
            save_figure=os.path.join(OUTPUT_DIR, f'shap_{name}.png'),
            show=False,
            mode='regression',
        )
        plt.close('all')

    print(f'Plots saved to: {os.path.abspath(OUTPUT_DIR)}')


if __name__ == '__main__':
    main()

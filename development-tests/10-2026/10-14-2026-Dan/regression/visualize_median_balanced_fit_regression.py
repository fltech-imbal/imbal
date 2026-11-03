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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_TRAIN_PATH = os.path.join(SCRIPT_DIR, 'sep_10mev_training_raw.csv')
RAW_TEST_PATH = os.path.join(SCRIPT_DIR, 'sep_10mev_testing_raw.csv')
TARGET = 'ln_peak_intensity'
RARE_THRESHOLD = np.log(10)
RANDOM_STATE = 42
SHAP_BACKGROUND_SIZE = 100


# Same feature plots and fixed legend locations as the SEP-C classification visualizer.
FEATURE_PLOTS = [
    ('Linear Speed', 'CME_DONKI_speed', 'feature_linear_speed.png', 'upper left'),
    ('Longitude', 'CME_DONKI_longitude', 'feature_longitude.png', 'upper left'),
    ('Latitude', 'CME_DONKI_latitude', 'feature_latitude.png', 'upper left'),
    ('Half Width', 'DONKI_half_width', 'feature_half_width.png', 'upper left'),
    ('Acceleration', 'Accelaration', 'feature_acceleration.png', 'upper right'),
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

    # Identify the same two speed-extreme errors for all five feature charts.
    speeds = pd.to_numeric(test['CME_DONKI_speed'], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(speeds).all():
        raise ValueError('Raw CME_DONKI_speed values must be finite.')
    fp_indices = np.flatnonzero(fp)
    fn_indices = np.flatnonzero(fn)
    highlighted_fp = (int(fp_indices[np.argmax(speeds[fp_indices])])
                      if len(fp_indices) else None)
    highlighted_fn = (int(fn_indices[np.argmin(speeds[fn_indices])])
                      if len(fn_indices) else None)
    for label, index in [('Highlighted FP (highest speed)', highlighted_fp),
                         ('Highlighted FN (lowest speed)', highlighted_fn)]:
        if index is None:
            print(f'{label}: none available')
        else:
            print(f'{label}: test row={index}, raw speed={speeds[index]:.6g}')

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
        # Hollow yellow rings keep the original black FP/FN triangles visible.
        for index in (highlighted_fp, highlighted_fn):
            if index is not None:
                ax.scatter([x[index]], [y_pred[index]], s=175, marker='o',
                           facecolors='none', edgecolors='#E5B900', linewidths=2.0,
                           label='Highlighted FP/FN' if index == (highlighted_fp if highlighted_fp is not None else highlighted_fn) else '_nolegend_', zorder=5)
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

    # Raw CSVs are used only for visualization; the model uses normalized features.
    raw_train = pd.read_csv(RAW_TRAIN_PATH)
    raw_test = pd.read_csv(RAW_TEST_PATH)
    for label, raw, normalized in [('training', raw_train, train), ('testing', raw_test, test)]:
        if len(raw) != len(normalized):
            raise ValueError(f'{label}: raw and normalized row counts differ.')
        if TARGET not in raw.columns or not np.allclose(
                pd.to_numeric(raw[TARGET], errors='raise').to_numpy(dtype=float),
                pd.to_numeric(normalized[TARGET], errors='raise').to_numpy(dtype=float),
                rtol=0, atol=1e-8, equal_nan=False):
            raise ValueError(f'{label}: raw and normalized target values do not align.')

    plot_feature_analysis(raw_test, y_test, y_pred)

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
            perplexity=240,
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
        # SHAP is calculated with normalized inputs, but the labels display
        # the original attribute names and values for the selected sample.
        raw_labels = []
        for feature in feature_names:
            raw_name = feature.removesuffix('_norm')
            if raw_name.startswith('log_'):
                raw_name = raw_name[4:]
            if raw_name == 'richardson_value':
                raw_name = 'half_richardson_value'
            if raw_name not in raw_test.columns:
                raise ValueError(f'No raw attribute corresponding to {feature}.')
            raw_value = raw_test.iloc[selected][raw_name]
            raw_labels.append(f'{raw_name} = {float(raw_value):.4g}'
                              if pd.notna(raw_value) else f'{raw_name} = NA')
        shap_explain_tabular_sample(
            sample=x_test[selected],
            model=model,
            training_data=background,
            feature_names=raw_labels,
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

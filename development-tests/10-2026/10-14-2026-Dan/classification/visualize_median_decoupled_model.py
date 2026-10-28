"""Visualize the saved median-F1 SEP-C classifier, including worst FP/FN SHAP."""
import json
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import imbal
from imbal.util.backend.explanation import shap_explain_tabular_sample

MODEL_PATH = 'saved_models/decoupled-fit-model-val.keras'
PARAMS_PATH = 'saved_models/median_params_decoupled_fit-val.json'
TRAIN_PATH = '../../../../tutorials/data/SEP-C/sep_10mev_training_classification.csv'
TEST_PATH = '../../../../tutorials/data/SEP-C/sep_10mev_testing_classification.csv'
OUTPUT_DIR = 'saved_results/median_model_visualizations'
TARGET = 'ln_peak_intensity'
RANDOM_STATE = 42
SHAP_BACKGROUND_SIZE = 100

# Original continuous targets are required for the Elevated distinction.
# This CSV must contain the same test samples in the same order as TEST_PATH.
CONTINUOUS_TEST_PATH = '../../../../tutorials/data/SEP-C/sep_10mev_testing.csv'
# Plot categories from the original continuous ln peak intensity.
# Background: y <= 0; Elevated: 0 < y < ln(10); SEP/Rare: y >= ln(10).
RARE_BOUNDARY = float(np.log(10.0))

# Feature names in the CSV; alternatives accommodate common naming conventions.
FEATURE_PLOT_GROUPS = [
    ('Linear Speed', ['CME_DONKI_speed_norm']),
    ('Longitude', ['CME_DONKI_longitude_norm']),
    ('Latitude', ['CME_DONKI_latitude_norm']),
    ('Half Width', ['DONKI_half_width_norm']),
    ('Acceleration', ['Accelaration_norm']),
]


def plot_feature_analysis(test, y_test, probabilities, predictions, threshold):
    """Create Griessler-style five-feature plots with a distinct Elevated band."""
    if not os.path.isfile(CONTINUOUS_TEST_PATH):
        raise FileNotFoundError(
            f'Continuous target CSV not found: {CONTINUOUS_TEST_PATH}. '
            'Set CONTINUOUS_TEST_PATH to the original regression test CSV; '
            'binary classification targets cannot identify Elevated samples.')
    continuous = pd.read_csv(CONTINUOUS_TEST_PATH)
    if TARGET not in continuous.columns:
        raise ValueError(f'{CONTINUOUS_TEST_PATH} does not contain {TARGET}.')
    if len(continuous) != len(test):
        raise ValueError('Continuous and classification test sets have different row counts.')
    # Prevent silent category misassignment due to different test-set ordering.
    feature_names = [c for c in test.columns if c != TARGET]
    missing = [c for c in feature_names if c not in continuous.columns]
    if missing:
        raise ValueError(f'Continuous CSV lacks classification features: {missing}')
    for name in feature_names:
        a = pd.to_numeric(test[name], errors='coerce').to_numpy(dtype=float)
        b = pd.to_numeric(continuous[name], errors='coerce').to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=True):
            raise ValueError(f'Test-set rows or feature values differ for {name}; '
                             'align the original continuous targets before plotting.')
    actual = pd.to_numeric(continuous[TARGET], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(actual).all():
        raise ValueError('Continuous targets must be finite.')
    rare = actual >= RARE_BOUNDARY
    elevated = (actual > 0.0) & (actual < RARE_BOUNDARY)
    background = actual <= 0.0
    if not np.array_equal(rare.astype(np.int32), y_test):
        raise ValueError('Continuous targets and classification labels disagree with '
                         'RARE_BOUNDARY/RARE_SIDE. Verify the class definition.')
    background = ~rare & ~elevated
    fp = (y_test == 0) & (predictions == 1)
    fn = (y_test == 1) & (predictions == 0)
    print(f'Plot categories: Background={background.sum()}, Elevated={elevated.sum()}, '
          f'Rare={rare.sum()}, FP={fp.sum()}, FN={fn.sum()}')

    resolved = []
    for title, candidates in FEATURE_PLOT_GROUPS:
        col = next((name for name in candidates if name in test.columns), None)
        if col is None:
            raise ValueError(f'Cannot find feature {title}. Available columns: {list(test.columns)}. '
                             'Update FEATURE_PLOT_GROUPS with your actual feature names.')
        resolved.append((title, col))

    # Fixed legend locations chosen from the reference figure (no automatic positioning).
    legend_locations = {
        'Linear Speed': 'upper left',
        'Longitude': 'upper left',
        'Latitude': 'upper left',
        'Half Width': 'upper left',
        'Acceleration': 'upper right',
    }
    output_names = {
        'Linear Speed': 'feature_linear_speed.png',
        'Longitude': 'feature_longitude.png',
        'Latitude': 'feature_latitude.png',
        'Half Width': 'feature_half_width.png',
        'Acceleration': 'feature_acceleration.png',
    }
    for title, column in resolved:
        x = pd.to_numeric(test[column], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError(f'Nonfinite values in {column}.')
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        # Correct predictions use colored circles; errors use black triangles only.
        for mask, label, color in [
            (background & ~fp, 'Background', 'blue'),
            (elevated & ~fp, 'Elevated', 'green'),
            (rare & ~fn, 'SEP', 'red'),
        ]:
            ax.scatter(x[mask], probabilities[mask], s=15, alpha=1.0,
                       c=color, marker='o', label=label, edgecolors='none', zorder=2)
        ax.scatter(x[fp], probabilities[fp], s=28, c='black', marker='^',
                   label='FP', edgecolors='none', zorder=4)
        ax.scatter(x[fn], probabilities[fn], s=28, c='black', marker='v',
                   label='FN', edgecolors='none', zorder=4)
        ax.axhline(threshold, color='gray', linestyle='--', linewidth=1.1)
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel('Predicted Rare probability')
        ax.set_ylim(-0.04, 1.04)
        ax.legend(loc=legend_locations[title], frameon=True, fontsize=9)
        fig.tight_layout()
        output = os.path.join(OUTPUT_DIR, output_names[title])
        fig.savefig(output, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {output}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PARAMS_PATH, encoding='utf-8') as f:
        params = json.load(f)
    threshold = params['best_decision_threshold']
    if threshold is None:
        raise ValueError('Median model has no saved best decision threshold.')

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    feature_names = list(train.drop(columns=[TARGET]).columns)
    x_train = train[feature_names].to_numpy(dtype=np.float32)
    x_test = test[feature_names].to_numpy(dtype=np.float32)
    y_test = test[TARGET].to_numpy(dtype=np.int32).ravel()
    if not np.isin(y_test, [0, 1]).all():
        raise ValueError('Expected binary classification labels (0 and 1).')

    # The model was saved as an imbal.classification.Model. Compile is unnecessary for inference.
    model = keras.models.load_model(MODEL_PATH, compile=False,
                                    custom_objects={'Model': imbal.classification.Model})
    probabilities = np.asarray(model.predict(x_test, verbose=0)).ravel()
    predictions = (probabilities >= threshold).astype(np.int32)
    false_positives = np.flatnonzero((y_test == 0) & (predictions == 1))
    false_negatives = np.flatnonzero((y_test == 1) & (predictions == 0))

    plot_feature_analysis(test, y_test, probabilities, predictions, threshold)

    # Compute confusion-matrix counts using imbal's own metric backend.
    from imbal.metrics.confusion_matrix import ConfusionMatrix
    counts = ConfusionMatrix.compute(
        variables_to_update=None,
        y_true=tf.constant(y_test.reshape(-1, 1), dtype=tf.float32),
        y_pred=tf.constant(predictions.reshape(-1, 1), dtype=tf.float32),
        dtype=tf.float32,
    )
    count_by_name = {key.value: int(value.numpy().item()) for key, value in counts.items()}
    matrix = np.array([
        [count_by_name['tn'], count_by_name['fp']],
        [count_by_name['fn'], count_by_name['tp']],
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap='Blues')
    ax.set_xticks([0, 1], labels=['Common (0)', 'Rare (1)'])
    ax.set_yticks([0, 1], labels=['Common (0)', 'Rare (1)'])
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('Actual class')
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha='center', va='center',
                color='white' if value > matrix.max() / 2 else 'black')
    ax.set_title(f'Median model confusion matrix (threshold={threshold:.4f})')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=200)
    plt.close(fig)
    print('Confusion matrix [rows=true, columns=predicted]:\n', matrix)

    # Visualize the entire test set using imbal's built-in representation t-SNE.
    # The default representation layer (-2) is the penultimate hidden layer.
    if len(x_test) >= 3:
        imbal.classification.tsne_visualization(
            model=model,
            data=x_test,
            labels=y_test,
            representation_layer_index=-2,
            perplexity=40,
            save_figure=os.path.join(OUTPUT_DIR, 'tsne_model.png'),
        )
        plt.close('all')
    else:
        print('Too few test samples for t-SNE.')

    # imbal's classification SHAP helper uses Kernel SHAP and a training background.
    rng = np.random.default_rng(RANDOM_STATE)
    background_indices = rng.choice(len(x_train),
                                    size=min(SHAP_BACKGROUND_SIZE, len(x_train)),
                                    replace=False)
    background = x_train[background_indices]
    # Select the most confidently wrong FP and FN (furthest past the decision threshold).
    for name, indices in [('fp', false_positives), ('fn', false_negatives)]:
        if not len(indices):
            print(f'No false {"positives" if name == "fp" else "negatives"} to explain.')
            continue
        if name == 'fp':
            selected = int(indices[np.argmax(probabilities[indices] - threshold)])
        else:
            selected = int(indices[np.argmax(threshold - probabilities[indices])])
        print(f'Worst {name.upper()}: test row={selected}, true={y_test[selected]}, '
              f'predicted={predictions[selected]}, probability={probabilities[selected]:.4f}, '
              f'distance from threshold={abs(probabilities[selected] - threshold):.4f}')
        # Call imbal's backend directly. The classification wrapper in this
        # version passes figure_save_path, which the backend does not accept.
        # The backend instead uses save_figure and accepts a single 1-D sample.
        shap_explain_tabular_sample(
            sample=x_test[selected],
            model=model,
            training_data=background,
            class_names=['Common', 'Rare'],
            feature_names=feature_names,
            actual_label=int(y_test[selected]),
            label_to_explain=int(predictions[selected]),
            plot_type='bar',
            save_figure=os.path.join(OUTPUT_DIR, f'shap_{name}.png'),
            show=False,
            mode='classification',
        )
        plt.close('all')

    print(f'Plots saved to: {os.path.abspath(OUTPUT_DIR)}')


if __name__ == '__main__':
    main()

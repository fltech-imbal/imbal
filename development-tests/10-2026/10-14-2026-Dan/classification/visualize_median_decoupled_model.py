"""Visualize the saved median-F1 SEP-C decoupled-fit classifier (no training)."""
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
    correct = np.flatnonzero(predictions == y_test)
    incorrect = np.flatnonzero(predictions != y_test)

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
            perplexity=min(30, len(x_test) - 1),
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
    for name, indices in [('correct', correct), ('incorrect', incorrect)]:
        if not len(indices):
            continue
        rare_indices = indices[y_test[indices] == 1]
        selected = int(rare_indices[0] if len(rare_indices) else indices[0])
        print(f'{name.title()} example: test row={selected}, true={y_test[selected]}, '
              f'predicted={predictions[selected]}, probability={probabilities[selected]:.4f}')
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

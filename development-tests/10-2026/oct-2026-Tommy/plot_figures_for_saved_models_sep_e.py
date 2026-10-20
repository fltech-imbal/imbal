import tensorflow as tf
from tools import plot_similarity_binned
import pandas as pd
import imbal
import numpy as np

DATA_PATH = "cleaned-dtw-SEP-EC-data"
DATA_PREFIX = 'sep_e_log_normalized'
OUTPUT_PATH = "paper-figures"

def load_sep_ec(path_prefix):
    training_data = pd.read_csv(path_prefix + '_training.csv')
    test_data = pd.read_csv(path_prefix + '_test.csv')
    val_data = pd.read_csv(path_prefix + '_validation.csv')

    training_labels = training_data.pop("p16.4_tplus6")
    val_labels = val_data.pop("p16.4_tplus6")
    test_labels = test_data.pop("p16.4_tplus6")
    training_data = training_data.to_numpy()
    val_data = val_data.to_numpy()
    test_data = test_data.to_numpy()
    training_labels = training_labels.to_numpy()
    val_labels = val_labels.to_numpy()
    test_labels = test_labels.to_numpy()
    return (training_data, training_labels), (val_data, val_labels), (test_data, test_labels)


(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_sep_ec(
    f"{DATA_PATH}/{DATA_PREFIX}",
)


def generate_plots(
    model_name,
    x_test,
    y_test,
):
    model = tf.keras.models.load_model(model_name)

    just_model_name = model_name.split('/')[-1].split('.')[0]

    predictions = model.predict(x_test)
    predictions = predictions.reshape(-1)
    y_test = y_test.reshape(-1)

    common_sample_mask = (y_test < np.log(10))
    common_predictions = predictions[common_sample_mask]
    rare_predictions = predictions[~common_sample_mask]
    common_labels = y_test[common_sample_mask]
    rare_labels = y_test[~common_sample_mask]

    mae = np.mean(np.abs(predictions - y_test))
    common_mae = np.mean(np.abs(common_predictions - common_labels))
    rare_mae = np.mean(np.abs(rare_predictions - rare_labels))

    imbal.regression.plot_true_vs_predictions(
        y_test,
        predictions,
        title=f'SEP-E - Common MAE: {common_mae:.4f}, Rare MAE: {rare_mae:.4f}, AORE: {(mae + rare_mae)/2:.4f}',
        save_figure=f"{OUTPUT_PATH}/{just_model_name}_tvp.png"
    )

    # imbal.regression.tsne_visualization(
    #     model,
    #     x_test,
    #     y_test,
    #     perplexity=150,
    #     representation_layer_index=-2,
    #     save_figure=f"{OUTPUT_PATH}/{just_model_name}_tsne.png"
    # )

    _, subset = imbal.regression.split(x_test, y_test, test_size=.33, seed=1)
    x_test, y_test = subset

    plot_similarity_binned(
        model,
        x_test[::3],
        y_test[::3],
        ticks=8,
        rank_based=False,
        save_figure=f"{OUTPUT_PATH}/{just_model_name}_similarity.png"
    )

MODEL_NAMES = [
    'models-final/sep_e_log_normalized_w_validation_distance_pcc_global_no_ratio5.keras',
    'models-final/sep_e_log_normalized_w_validation_entropy_global_2.keras',
    'models-final/sep_e_log_normalized_w_validation_distance_difference_1.keras'
]

import gc

for name in MODEL_NAMES:
    generate_plots(
        name,
        x_test,
        y_test
    )
    gc.collect()
import tensorflow as tf
import pandas as pd
import imbal
import keras

import numpy as np
OUTPUT_PATH = "results"
DATA_PATH = "cleaned-dtw-SEP-EC-data"
DATA_PREFIX = 'sep_e_log_normalized'
FULL_NAME = '_decoupled_w_validation_ae_3'
USE_DELTA = False
WEIGHT_CANDIDATES = False
INCLUDE_IDS = True
SAVE = True

# def pcc(y_true, y_pred):
#     y_true = tf.reshape(y_true, tf.shape(y_pred))
#
#     y_true_centered = y_true - tf.reduce_mean(y_true)
#     y_pred_centered = y_pred - tf.reduce_mean(y_pred)
#
#     return 1 - (
#         tf.reduce_sum(y_true_centered * y_pred_centered) /
#         (tf.norm(y_true_centered) * tf.norm(y_pred_centered))
#     )
#
# def loss_fn(y_true, y_pred):
#     return keras.losses.mean_squared_error(y_true, y_pred) + 0.5*pcc(y_true, y_pred)


# Load the entire model (architecture, weights, and optimizer state)
model = tf.keras.models.load_model(
    f'models/{DATA_PREFIX}{FULL_NAME}.keras',
    custom_objects={
        # 'loss_fn': loss_fn,
        'Model' : imbal.regression.Model
    }
)
print(type(model))

# Verify the model structure
model.summary()


def load_sep_ec(path_prefix):
    training_data = pd.read_csv(path_prefix + '_training.csv')
    test_data = pd.read_csv(path_prefix + '_test.csv')
    val_data = pd.read_csv(path_prefix + '_validation.csv')
    columns = training_data.columns
    if USE_DELTA:
        training_labels = training_data.pop("delta_log_Intensity")
        val_labels = val_data.pop("delta_log_Intensity")
        test_labels = test_data.pop("delta_log_Intensity")
    else:
        training_labels = training_data.pop("p16.4_tplus6")
        val_labels = val_data.pop("p16.4_tplus6")
        test_labels = test_data.pop("p16.4_tplus6")

    training_ids = training_data.pop('Event ID') if INCLUDE_IDS else None
    val_ids = val_data.pop('Event ID') if INCLUDE_IDS else None
    test_ids = test_data.pop('Event ID') if INCLUDE_IDS else None

    training_data = training_data.to_numpy()
    val_data = val_data.to_numpy()
    test_data = test_data.to_numpy()
    training_labels = training_labels.to_numpy()
    val_labels = val_labels.to_numpy()
    test_labels = test_labels.to_numpy()
    return columns, (training_data, training_labels, training_ids), (val_data, val_labels, val_ids), (test_data, test_labels, test_ids)

columns, (x_train, y_train, train_ids), (x_val, y_val, val_ids), (x_test, y_test, test_ids) = load_sep_ec(
    f"{DATA_PATH}/{DATA_PREFIX}{'_w_ids' if INCLUDE_IDS else ''}",
)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

print(y_train[y_train >= np.log(10)].shape)
print(y_train[y_train < np.log(10)].shape)
print(y_val[y_val >= np.log(10)].shape)
print(y_val[y_val < np.log(10)].shape)
print(y_test[y_test >= np.log(10)].shape)
print(y_test[y_test < np.log(10)].shape)

predictions = model.predict(x_test)
predictions = predictions.reshape(-1)
y_test = y_test.reshape(-1)

common_sample_mask = (y_test > -0.5) & (y_test < 0.5) if False else (y_test < np.log(10))
common_predictions = predictions[common_sample_mask]
rare_predictions = predictions[~common_sample_mask]
common_labels = y_test[common_sample_mask]
rare_labels = y_test[~common_sample_mask]

mae = np.mean(np.abs(predictions - y_test))
common_mae = np.mean(np.abs(common_predictions - common_labels))
rare_mae = np.mean(np.abs(rare_predictions - rare_labels))

print(model.best_weight_index)

import matplotlib.pyplot as plt

dates_map = {
    10 : "01-22-2012",
    22 : "08-31-2012",
    27 : "05-21-2013",
    30 : "01-06-2014",
    32 : "02-19-2014",
    36 : "11-01-2014",
    41 : "12-31-2015",
    43 : "09-04-2017"
}

def plot_true_vs_predictions(
    labels,
    predictions,
    x_axis_label=None,
    y_axis_label=None,
    title=None,
    color=None,
    marker=None,
    size=None,
    save_figure=None,
    ids=None
):
    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)

    data_min = np.min([labels.min(), predictions.min()]) - 1
    data_max = np.max([labels.max(), predictions.max()]) + 1

    # Create comparison plot
    plt.figure(figsize=(7, 6))
    plt.plot([data_min, data_max], [data_min, data_max], linestyle="--", linewidth=1, color='black')
    if ids is None:
        plt.scatter(labels, predictions, color="#00FF0044" if color is None else color, s=size, marker=marker)
    else:
        for value in ids.unique():
            plt.scatter(labels[ids == value], predictions[ids == value], s=size, marker=marker, label=dates_map[value])
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.xlabel("True Label" if x_axis_label is None else x_axis_label)
    plt.ylabel("Predicted Label" if y_axis_label is None else y_axis_label)
    plt.xlim(data_min, data_max)
    plt.ylim(data_min, data_max)
    if title is not None:
        plt.title(title)
    plt.legend()
    if save_figure is not None:
        plt.savefig(save_figure)

    plt.show()

plot_true_vs_predictions(
    y_test,
    predictions,
    title=f'SEP-E - Common MAE: {common_mae:.4f}, Rare MAE: {rare_mae:.4f}, AORE: {(mae + rare_mae)/2:.4f}{f", Alpha: {[0.1*(i+1) for i in range(10)][model.best_weight_index]:.1f}" if WEIGHT_CANDIDATES else ""}',
    save_figure=f"{OUTPUT_PATH}/{DATA_PREFIX}{FULL_NAME}_multicolor.png" if SAVE else None,
    ids=test_ids if INCLUDE_IDS else None,
    size=3 if INCLUDE_IDS else None,
)

imbal.regression.tsne_visualization(
    model,
    x_test,
    y_test,
    # perplexity=300,
    save_figure=f'{OUTPUT_PATH}/{DATA_PREFIX}{FULL_NAME}-sep-proton-time-series-tsne.png'
)

columns = columns.tolist()

rare_samples_mispredicted = x_test[(y_test > np.log(10)) & (predictions < np.log(10))]
rare_label_mispredicted = y_test[(y_test > np.log(10)) & (predictions < np.log(10))]
rare_predictions_mispredicted = predictions[(y_test > np.log(10)) & (predictions < np.log(10))]

misprediction_errors = np.abs(rare_predictions_mispredicted - rare_label_mispredicted)
error_indices = np.argsort(misprediction_errors)

high_error_rare_sample = rare_samples_mispredicted[error_indices[-1]]

imbal.regression.shap_explain_tabular_sample(
    sample=high_error_rare_sample,
    model=model,
    training_data=x_train,
    feature_names=columns,
)

# imbal.regression.lime_explain_tabular_sample(
#     x_test[common_sample_mask][0],
#     model,
#     x_train,
#     feature_names=columns,
#     actual_label=round(float(y_test[common_sample_mask][0]), 3),
#     figure_save_path='common.html'
# )
#
# imbal.regression.lime_explain_tabular_sample(
#     x_test[~common_sample_mask][0],
#     model,
#     x_train,
#     feature_names=columns,
#     actual_label=round(float(y_test[~common_sample_mask][0]), 3),
#     figure_save_path='rare.html'
# )
#
# common_error_indices = np.argsort(np.abs(common_predictions - common_labels))
# print(common_error_indices[-5:])
#
# imbal.regression.lime_explain_tabular_sample(
#     x_test[common_sample_mask][common_error_indices[-1]],
#     model,
#     x_train,
#     feature_names=columns,
#     actual_label=round(float(y_test[common_sample_mask][common_error_indices[-1]]), 3),
#     figure_save_path='high_error.html'
# )
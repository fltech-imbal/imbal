import tensorflow as tf
import pandas as pd
import imbal
import keras

import numpy as np
DATA_PATH = 'sep_ec_log_normalized'

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
    'models/sep_ec_log_normalized_decoupled_w_validation_ae_denseweight.keras',
    custom_objects={
        # 'loss_fn': loss_fn,
        'Model' : imbal.regression.Model
    }
)

# Verify the model structure
model.summary()

def load_sep_ec(path_prefix):
    training_data = pd.read_csv(path_prefix + '_training.csv')
    test_data = pd.read_csv(path_prefix + '_test.csv')
    val_data = pd.read_csv(path_prefix + '_validation.csv')
    training_labels = training_data.pop("delta_log_Intensity")
    val_labels = val_data.pop("delta_log_Intensity")
    test_labels = test_data.pop("delta_log_Intensity")
    columns = training_data.columns
    training_data = training_data.to_numpy()
    val_data = val_data.to_numpy()
    test_data = test_data.to_numpy()
    training_labels = training_labels.to_numpy()
    val_labels = val_labels.to_numpy()
    test_labels = test_labels.to_numpy()
    return (training_data, training_labels), (val_data, val_labels), (test_data, test_labels), columns

(x_train, y_train), (x_val, y_val), (x_test, y_test), columns = load_sep_ec(
    f"SEP-E/{DATA_PATH}",
)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

predictions = model.predict(x_test)
predictions = predictions.reshape(-1)
y_test = y_test.reshape(-1)

common_sample_mask = (y_test > -0.5) & (y_test < 0.5)
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
    title=f'Common MAE: {common_mae:.4f}, Rare MAE: {rare_mae:.4f}, AORE: {(mae + rare_mae)/2:.4f}',
    save_figure='sep-proton-time-series-true-vs-predicted.png'
)

imbal.regression.tsne_visualization(
    model,
    x_test,
    y_test,
    # perplexity=300,
    save_figure='sep-proton-time-series-tsne.png'
)

columns = columns.tolist()

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

common_error_indices = np.argsort(np.abs(common_predictions - common_labels))
print(common_error_indices[-5:])

imbal.regression.lime_explain_tabular_sample(
    x_test[common_sample_mask][common_error_indices[-1]],
    model,
    x_train,
    feature_names=columns,
    actual_label=round(float(y_test[common_sample_mask][common_error_indices[-1]]), 3),
    figure_save_path='high_error.html'
)
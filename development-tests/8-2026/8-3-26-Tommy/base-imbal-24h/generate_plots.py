import tensorflow as tf
import pandas as pd
import imbal
import keras

import numpy as np
OUTPUT_PATH = "results"
DATA_PATH = "cleaned-dtw-SEP-EC-data"
DATA_PREFIX = 'sep_e_log_normalized'
UNNORMALIZED_DATA_PREFIX = 'sep_e_log'
FULL_NAME = '_decoupled_w_validation_ae_3'
USE_DELTA = False
WEIGHT_CANDIDATES = False
INCLUDE_IDS = True
INCLUDE_TIMES = True
SAVE = True

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

    training_times = training_data.pop('Timestamp') if INCLUDE_TIMES else None
    val_times = val_data.pop('Timestamp') if INCLUDE_TIMES else None
    test_times = test_data.pop('Timestamp') if INCLUDE_TIMES else None

    return columns, (training_data, training_labels, training_ids, training_times), (val_data, val_labels, val_ids, val_times), (test_data, test_labels, test_ids, test_times)

columns, (x_train, y_train, train_ids, train_times), (x_val, y_val, val_ids, val_times), (x_test, y_test, test_ids, test_times) = load_sep_ec(
    f"{DATA_PATH}/{DATA_PREFIX}{'_w_ids' if INCLUDE_IDS else ''}{'_w_times' if INCLUDE_TIMES else ''}",
)

x_train_df = x_train
y_train_df = y_train
x_val_df = x_val
y_val_df = y_val
x_test_df = x_test
y_test_df = y_test

x_train = x_train.to_numpy()
x_val = x_val.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
y_test = y_test.to_numpy()

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

# # True vs predicted - colored by time series
# plot_true_vs_predictions(
#     y_test,
#     predictions,
#     title=f'SEP-E - Common MAE: {common_mae:.4f}, Rare MAE: {rare_mae:.4f}, AORE: {(mae + rare_mae)/2:.4f}{f", Alpha: {[0.1*(i+1) for i in range(10)][model.best_weight_index]:.1f}" if WEIGHT_CANDIDATES else ""}',
#     save_figure=f"{OUTPUT_PATH}/{DATA_PREFIX}{FULL_NAME}_multicolor.png" if SAVE else None,
#     ids=test_ids if INCLUDE_IDS else None,
#     size=3 if INCLUDE_IDS else None,
# )
#
# # TSNE
# imbal.regression.tsne_visualization(
#     model,
#     x_test,
#     y_test,
#     # perplexity=300,
#     save_figure=f'{OUTPUT_PATH}/{DATA_PREFIX}{FULL_NAME}-sep-proton-time-series-tsne.png'
# )

columns = columns.tolist()

rare_samples_mispredicted = x_test[(y_test > np.log(10))]
rare_label_mispredicted = y_test[(y_test > np.log(10))]
rare_predictions_mispredicted = predictions[(y_test > np.log(10))]

misprediction_errors = np.abs(rare_predictions_mispredicted - rare_label_mispredicted)
error_indices = np.argsort(misprediction_errors)

high_error_rare_sample = rare_samples_mispredicted[error_indices[-1]]
high_error_rare_label = rare_label_mispredicted[error_indices[-1]]

(x_sub_1, y_sub_1), (x_sub_2, y_sub_2) = imbal.regression.split(x_train, y_train, train_size=0.03)

# imbal.regression.shap_explain_tabular_sample(
#     sample=high_error_rare_sample,
#     actual_label=round(high_error_rare_label, 3),
#     model=model,
#     training_data=x_sub_1,
#     feature_names=columns,
#     save_figure=f"{OUTPUT_PATH}/{DATA_PREFIX}{FULL_NAME}_shap_explain_rare_high_error_sample.png" if SAVE else None,
# )
#
# common_samples_mispredicted = x_test[(y_test < np.log(10))]
# common_label_mispredicted = y_test[(y_test < np.log(10))]
# common_predictions_mispredicted = predictions[(y_test < np.log(10))]
#
# misprediction_errors = np.abs(common_predictions_mispredicted - common_label_mispredicted)
# error_indices = np.argsort(misprediction_errors)
#
# high_error_common_sample = common_samples_mispredicted[error_indices[-1]]
# high_error_common_label = common_label_mispredicted[error_indices[-1]]
#
# imbal.regression.shap_explain_tabular_sample(
#     sample=high_error_common_sample,
#     actual_label=round(high_error_common_label, 3),
#     model=model,
#     training_data=x_sub_1,
#     feature_names=columns,
#     save_figure=f"{OUTPUT_PATH}/{DATA_PREFIX}{FULL_NAME}_shap_explain_common_high_error_sample.png" if SAVE else None,
# )

import matplotlib.dates as mdates
def plot_time_series(
    df,
    predictions,
    labels=None,
    save_figure=None
):
    fig, ax = plt.subplots(figsize=(20, 6))

    # ax.plot(
    #     df["Timestamp"],
    #     labels,
    #     label="Actual Proton",
    #     linewidth=2
    # )

    ax.plot(
        df["Timestamp"],
        df['e0.5_t'],
        label="E0.5 Channel",
        linewidth=2,
        color="#FF8800"
    )

    ax.plot(
        df["Timestamp"],
        df['e1.8_t'],
        label="E1.8 Channel",
        linewidth=2,
        color="#FF4400"
    )

    ax.plot(
        df["Timestamp"],
        df['e4.4_t'],
        label="E4.4 Channel",
        linewidth=2,
        color="#FF0000"
    )

    ax.plot(
        df["Timestamp"],
        df['p6.1_t'],
        label="P6.1 Channel",
        linewidth=2,
        color="#0044FF"
    )

    ax.plot(
        df["Timestamp"],
        df['p16.4_t'],
        label="P16.4 Channel",
        linewidth=2,
        color="#000000" # 00CCCC
    )

    ax.plot(
        df["Timestamp"],
        df['p33.0_t'],
        label="P33.0 Channel",
        linewidth=2,
        color="#00DD00"
    )

    if labels is not None:
        ax.plot(
            df["Timestamp"],
            labels,
            label="Actual label",
            linewidth=2,
            color="#000000"
        )

    ax.plot(
        df["Timestamp"],
        df['p16.4_t'],
        #label="P16.4 Channel",
        linewidth=2,
        color="#000000" # 00CCCC
    )

    ax.plot(
        df["Timestamp"] + pd.Timedelta(minutes=30),
        predictions,
        label="Predicted P16.4 Channel",
        linewidth=2,
        color="#8800FF"
    )



    plt.ylim(bottom=-10)


    # ax.plot(
    #     df["Timestamp"],
    #     df["predicted_proton_posner"],
    #     label="Predicted proton (Posner method)",
    #     linewidth=2
    # )
    #
    # ax.plot(
    #     df["Timestamp"],
    #     df["predicted_proton_m1"],
    #     label="Predicted proton (M1 method)",
    #     linewidth=2
    # )
    #
    # ax.plot(
    #     df["Timestamp"],
    #     df["electron"],
    #     label="Electron",
    #     linewidth=2
    # )
    #
    # ax.plot(
    #     df["Timestamp"],
    #     df["high_energy_electron"],
    #     label="High-energy electron",
    #     linewidth=2
    # )

    ax.axhline(
        y=np.log(10),
        color="black",
        linestyle="--",
        linewidth=1.5
    )

    #ax.set_xlabel("Event on 2001-04-15", fontsize=16)
    #ax.set_ylabel("ln(Flux (cc/s/sr))", fontsize=16)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=90, fontsize=12)
    ax.tick_params(axis="both", labelsize=12)

    ax.legend(
        loc="upper left",
        fontsize=11
    )
    fig.tight_layout()

    if save_figure is not None:
        plt.savefig(save_figure)

    plt.show()

columns, (x_train, y_train, train_ids, train_times), (x_val, y_val, val_ids, val_times), (x_test, y_test, test_ids, test_times) = load_sep_ec(
    f"{DATA_PATH}/{UNNORMALIZED_DATA_PREFIX}{'_w_ids' if INCLUDE_IDS else ''}{'_w_times' if INCLUDE_TIMES else ''}",
)

x_train_df = x_train
y_train_df = y_train
x_val_df = x_val
y_val_df = y_val
x_test_df = x_test
y_test_df = y_test

x_train = x_train.to_numpy()
x_val = x_val.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
y_test = y_test.to_numpy()

test_times = pd.to_datetime(test_times)
x_test_df = pd.concat([x_test_df, test_times], axis=1)
x_test_df = x_test_df.sort_values("Timestamp")

print(x_test_df)

print(x_test_df.shape)
print(test_ids.shape)
print(y_test_df.shape)
plot_time_series(
    x_test_df[test_ids==41],
    predictions[test_ids==41],
    save_figure=f"{OUTPUT_PATH}/{DATA_PREFIX}{FULL_NAME}_high_error_time_series.png" if SAVE else None,
)
plot_time_series(
    x_test_df[test_ids==10],
    predictions[test_ids==10],
    save_figure=f"{OUTPUT_PATH}/{DATA_PREFIX}{FULL_NAME}_low_error_time_series.png" if SAVE else None,
)
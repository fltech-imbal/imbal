import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import layers
import os
from aore_metric import AORE
import matplotlib.pyplot as plt

import imbal

seed = 42
tf.keras.utils.set_random_seed(
    seed
)

target_column = "ln_peak_intensity"

max_epochs = 500
batch_size = 32

# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv("sep_10mev_training_pseudo_labeled.csv")
test_data  = pd.read_csv("sep_10mev_testing_pseudo_labeled.csv")

y_train = train_data[target_column].values.reshape(-1, 1).astype("float32")
y_test  = test_data[target_column].values.reshape(-1, 1).astype("float32")

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test  = test_data.drop(columns=[target_column]).values.astype(np.float32)

# ----------------------------
# Model
# ----------------------------
def build_model(input_shape: int) -> imbal.regression.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
    hidden2 = layers.Dense(12, activation="relu", name="hidden_layer2")(hidden1)
    hidden3 = layers.Dense(8, activation="relu", name="hidden_layer3")(hidden2)
    hidden4 = layers.Dense(6, activation="relu", name="hidden_layer4")(hidden3)
    flatten = layers.Flatten()(hidden4)
    outputs = layers.Dense(1, name="output_layer")(flatten)
    built_model = imbal.regression.Model(inputs=inputs, outputs=outputs, name="sep_model")
    return built_model

MODEL_SAVE_PATH = "saved_models/balanced-fit-model-val-ae.keras"
LOAD_SAVED_MODEL = True

if LOAD_SAVED_MODEL and os.path.exists(MODEL_SAVE_PATH):
    print(f'Loading saved regression model from {MODEL_SAVE_PATH}')
    model = keras.models.load_model(
        MODEL_SAVE_PATH,
        custom_objects={'Model': imbal.regression.Model,
                        'AORE': AORE,}
    )
else:
    model = build_model(x_train.shape[1])

    # ----------------------------
    # Validation Set
    # ----------------------------
    labels_kde = y_train.reshape(-1).copy()
    kde = imbal.regression.fit_kde(labels_kde)
    densities = imbal.regression.get_sample_densities(labels_kde, kde)

    # Comment the below out if using the explore alphas version of the call
    # sample_weights = imbal.regression.generate_sample_weights(densities)
    # (x_train, y_train, sw), (x_val, y_val, sw_val) =  imbal.regression.split(x_train, y_train, sample_weights=sample_weights, test_size=0.2)

    # ----------------------------
    # Training
    # ----------------------------

    model.compile(
        loss="mean_squared_error",
        weighted_metrics=[AORE(threshold=np.log(10)), "mae"],
        optimizer="adam",
        generate_decoder_branch=True,
    )

    PATIENCE = 50

    # model.balanced_fit(
    #     x_train,
    #     y_train,
    #     validation_data = (x_val, y_val.reshape(-1, 1), sw_val),
    #     sample_weight=sw,
    #     batch_size=batch_size,
    #     epochs=max_epochs,
    #     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)]
    # )

    # Uncomment the below if you want to try exploring different alpha values
    from imbal.regression import reciprocal_importance
    alpha_candidates = [0.2, 0.5, 0.8, 0.9, 1.0, 1.1]
    weight_candidates = reciprocal_importance(densities, alpha=alpha_candidates)
    (x_train, y_train, sw_candidates), (x_val, y_val, sw_val) =  imbal.regression.split(x_train, y_train, sample_weights=weight_candidates, test_size=0.2)
    candidate_evaluation_weights = np.ones(len(y_val))

    model.balanced_fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val.reshape(-1, 1), sw_val),
        candidate_evaluation_sample_weight=candidate_evaluation_weights,
        sample_weight=sw_candidates,
        batch_size=batch_size,
        epochs=max_epochs,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)]
    )

    best_alpha_index = int(model.best_weight_index)
    best_alpha = float(alpha_candidates[best_alpha_index])

    model.save(MODEL_SAVE_PATH)

    import json

    with open("saved_models/best_params_balanced_fit_regression-val-ae.json", "w") as f:
        json.dump({
            "best_alpha_index": best_alpha_index,
            "best_alpha": best_alpha
        }, f, indent=4)

# ----------------------------
# Evaluation
# ----------------------------
results = model.evaluate(x_test, y_test)
loss, aore, mae = results
predictions = model.predict(x_test)

print(f"Test Loss: {loss:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"AORE: {aore:.4f}")

threshold = np.log(10)

y_true = y_test.reshape(-1)
y_pred = predictions.reshape(-1)

common_mask = y_true < threshold
rare_mask = y_true >= threshold

common_mae = np.mean(np.abs(y_true[common_mask] - y_pred[common_mask]))
rare_mae = np.mean(np.abs(y_true[rare_mask] - y_pred[rare_mask]))

print(f"Common sample MAE (< ln(10)): {common_mae:.4f}")
print(f"Rare sample MAE (>= ln(10)): {rare_mae:.4f}")


# ----------------------------
# Visualization
# ----------------------------
imbal.regression.plot_true_vs_predictions(
    y_test,
    predictions
)

# ----------------------------
# Visualization
# ----------------------------
# ----------------------------
# Visualization
# ----------------------------
def plot_prediction_feature_graphs(
    dataframe,
    y_true,
    y_pred,
    output_path="predicted_peak_intensity_feature_plots.png",
):
    """Plot predicted peak intensity against CME speed, longitude, and latitude.

    Marker definitions:
      - Blue circles: background samples (actual value < 0)
      - Green circles: non-background non-SEP events (0 <= actual value < ln(10))
      - Red circles: SEP samples (actual value >= ln(10))
      - Black upward triangles: false positives
      - Black downward triangles: false negatives

    Replace the dictionary keys below if your CSV uses different column names.
    """
    threshold = np.log(10)

    background_mask = y_true < 0
    non_background_non_sep_mask = (y_true >= 0) & (y_true < threshold)
    sep_mask = y_true >= threshold

    false_positive_mask = background_mask & (y_pred >= threshold)
    false_negative_mask = sep_mask & (y_pred < threshold)

    # Exclude FP/FN samples from the ordinary circle markers so each point is
    # shown only once and the triangle markers remain easy to see.
    ordinary_background_mask = background_mask & ~false_positive_mask
    ordinary_sep_mask = sep_mask & ~false_negative_mask

    feature_settings = [
        ("CME_DONKI_speed_norm", "Linear Speed"),
        ("CME_DONKI_longitude_norm", "Longitude"),
        ("CME_DONKI_latitude_norm", "Latitude"),
    ]

    missing_columns = [
        column_name
        for column_name, _ in feature_settings
        if column_name not in dataframe.columns
    ]
    if missing_columns:
        raise KeyError(
            "The following plotting columns were not found in the test CSV: "
            + ", ".join(missing_columns)
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for axis, (column_name, display_name) in zip(axes, feature_settings):
        x_values = dataframe[column_name].to_numpy()

        axis.scatter(
            x_values[ordinary_background_mask],
            y_pred[ordinary_background_mask],
            c="blue",
            marker="o",
            s=28,
            label="Background",
            zorder=1,
        )
        axis.scatter(
            x_values[non_background_non_sep_mask],
            y_pred[non_background_non_sep_mask],
            c="green",
            marker="o",
            s=32,
            label="Elevated",
            zorder=2,
        )
        axis.scatter(
            x_values[ordinary_sep_mask],
            y_pred[ordinary_sep_mask],
            c="red",
            marker="o",
            s=35,
            label="SEP",
            zorder=3,
        )
        axis.scatter(
            x_values[false_positive_mask],
            y_pred[false_positive_mask],
            c="black",
            marker="^",
            s=55,
            label="FP",
            zorder=4,
        )
        axis.scatter(
            x_values[false_negative_mask],
            y_pred[false_negative_mask],
            c="black",
            marker="v",
            s=55,
            label="FN",
            zorder=4,
        )

        axis.set_title(f"Predicted Peak Intensity LN vs {display_name}")
        axis.set_xlabel(display_name)
        axis.set_ylabel("Predicted Peak Intensity LN")
        axis.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved prediction feature plots to: {output_path}")
    print(f"False positive count: {np.sum(false_positive_mask)}")
    print(f"False negative count: {np.sum(false_negative_mask)}")


plot_prediction_feature_graphs(test_data, y_true, y_pred)

# ----------------------------
# False Positive Analysis
# ----------------------------

false_positive_mask = (y_true < 0) & (y_pred >= np.log(10))
false_positive_count = np.sum(false_positive_mask)

print(f"False positive count: {false_positive_count}")

# Replace these with the actual column names you want.
columns_to_print = [
    "CME_DONKI_speed_norm",
    "CME_DONKI_latitude_norm",
    "CME_DONKI_longitude_norm",
]

print("\nData from False Positive Samples:")
print(test_data.loc[false_positive_mask, columns_to_print])
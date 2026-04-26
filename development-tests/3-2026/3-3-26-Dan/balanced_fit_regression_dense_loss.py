import time
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import layers

import imbal
from imbal.util.backend.constants import ModelType
from modular_cross_validation import (
    fit_k_folds_modular,
    BalancedFitParams,
    BalancedFitStrategy,
)
from regression_metrics import compute_regression_metrics
from sep_regression_plot import plot_predicted_vs_actual
from custom_metrics import AORE, RareMAE, RegressionFalsePositives, RegressionFalseNegatives


# ----------------------------
# Config
# ----------------------------
seed = 67
tf.keras.utils.set_random_seed(
    seed
)

target_column = "ln_peak_intensity"
threshold = np.log(10.0)

num_folds_for_split = 5
max_epochs = 1000
batch_size = 32

# EarlyStopping rule:
early_stop_patience = 50
local_minima_patience = 30
early_stop_min_delta = 0.001  # e.g. 1e-4 for stricter improvements
local_minima_min_delta = 0.001  # e.g. 1e-4 for stricter improvements
monitor_metric = "val_aore"


# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_training.csv")
test_data  = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_testing.csv")

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
    outputs = layers.Dense(1, name="output_layer")(hidden4)
    built_model = imbal.regression.Model(inputs=inputs, outputs=outputs, name="one_hidden_layer_6_units")
    return built_model


def build_model_from_research_paper(input_shape: int) -> imbal.regression.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
    hidden2 = layers.Dense(9, activation="relu", name="hidden_layer2")(hidden1)
    hidden3 = layers.Dense(6, activation="relu", name="hidden_layer3")(hidden2)
    outputs = layers.Dense(1, name="output_layer")(hidden3)
    built_model = imbal.regression.Model(inputs=inputs, outputs=outputs, name="one_hidden_layer_6_units")
    return built_model


# ----------------------------
# Train with EarlyStopping on val_loss, and vary class weights
# ----------------------------
import random
import os

def reset_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
all_results = []

reset_seeds(seed)

model = build_model(x_train.shape[1])
#model = build_model_from_research_paper(x_train.shape[1])

# labels_kde = y_train.reshape(-1).copy()
# kde = imbal.regression.fit_kde(labels_kde)
# densities = imbal.regression.get_sample_densities(labels_kde, kde)
# sample_weights = denseweight_sample_weights(densities, alpha, 1e-4)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=early_stop_min_delta,
    patience=early_stop_patience,
    mode="min",
    restore_best_weights=True,
    verbose=1,
)

start_cpu = time.process_time()

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        RareMAE(threshold=threshold, name="mae_rare"),
        AORE(threshold=threshold, name="aore"),
        RegressionFalsePositives(threshold=threshold, name="fp"),
        RegressionFalseNegatives(threshold=threshold, name="fn"),
    ],
)

# --- modular CV setup ---
strategy = BalancedFitStrategy()

params = BalancedFitParams(
    x=x_train,
    y=y_train,
    #sample_weight=sample_weights,
    batch_size=batch_size,
    shuffle=True,                # keep your intended behavior
    stratify_batches=True,        # matches your current call
    callbacks=[early_stop],       # NOTE: only pass the EarlyStopping here if you want it cloned per fold
    kwargs={

    },
)

history, model, selection_info = fit_k_folds_modular(
    model=model,
    x=x_train,
    y=y_train,
    strategy=strategy,
    params=params,
    batch_size=batch_size,                 # driver-level override (kept consistent)
    num_folds=num_folds_for_split,
    shuffle=True,
    seed=seed,
    metric=monitor_metric,
    mode=ModelType.REGRESSION,
    alpha_candidates=alpha_values,
    alpha_eps=1e-4,
)


end_cpu = time.process_time()

results = model.evaluate(x_test, y_test, verbose=0)

loss, mae = results
print(f"\nCPU time: {end_cpu - start_cpu:.4f} sec")
print(f"loss: {loss:.4f}")
print(f"mae: {mae:.4f}")

y_pred = model.predict(x_test, verbose=0)

metrics_tuple = compute_regression_metrics(
    y_true=y_test,
    y_pred=y_pred,
    threshold=np.log(10.0),
)
overall_mae, rare_mae, _, aore, overall_pcc, rare_pcc, aorc, fp, fn = metrics_tuple

all_results.append({
    "alpha": selection_info["selected_alpha"],
    "loss": loss,
    "mean_val_loss": selection_info["mean_best_val_loss"],
    "overall_mae": overall_mae,
    "rare_mae": rare_mae,
    "aore": aore,
    "overall_pcc": overall_pcc,
    "rare_pcc": rare_pcc,
    "aorc": aorc,
    "FP": fp,
    "FN": fn,
    "epochs": len(history.history["loss"]),
})

# ----------------------------------
# Summary table
# ----------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.precision', 4)
results_df = pd.DataFrame(all_results)
print("\n=== Sweep Summary ===")
print(results_df)

# -------------- Regression Plot ---------------
def run_regression_plot(data, target, threshold_regplot, model_sep_regplot, x_test_regplot):
    y_true = data[target].to_numpy()
    y_pred = model_sep_regplot.predict(x_test_regplot, batch_size=512, verbose=0).reshape(-1)

    plot_predicted_vs_actual(
        y_true=y_true,
        y_pred=y_pred,
        threshold=threshold_regplot,
        out_png="pred_vs_actual_ln_peak_balanced_fit_current_model.png",
        title="Predicted vs Actual ln(peak intensity) (Current Model)",
        show=False
    )

    # plot_predicted_vs_actual(
    #     y_true=y_true,
    #     y_pred=y_pred,
    #     threshold=threshold_regplot,
    #     out_png="pred_vs_actual_ln_peak_balanced_fit_paper_model.png",
    #     title="Predicted vs Actual ln(peak intensity) (Paper Model)",
    #     show=False
    # )


run_regression_plot(test_data, target_column, threshold, model, x_test)

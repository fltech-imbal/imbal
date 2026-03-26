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
    DecoupledFitStageOneParams,
    DecoupledFitStageTwoParams,
    DecoupledFitStageOneStrategy,
    DecoupledFitStageTwoExtraLayerStrategy,
)
from sep_regression_plot import plot_predicted_vs_actual
from custom_callbacks import ConvergenceStopping
from custom_metrics import AORE, RareMAE, RegressionFalsePositives, RegressionFalseNegatives, RegressionF1Score
from regression_metrics import compute_regression_metrics


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
monitor_metric = "val_loss"


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


alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
all_results = []

stage_one_model = build_model(x_train.shape[1])
#model = build_model_from_research_paper(x_train.shape[1])

labels_kde = y_train.reshape(-1).copy()
kde = imbal.regression.fit_kde(labels_kde)
densities = imbal.regression.get_sample_densities(labels_kde, kde)
sample_weights = imbal.regression.generate_sample_weights(densities)

# early_stop = keras.callbacks.EarlyStopping(
#     monitor="val_loss",
#     min_delta=early_stop_min_delta,
#     patience=early_stop_patience,
#     mode="max",
#     restore_best_weights=True,
#     verbose=1,
# )

stage_one_convergence_stop = ConvergenceStopping("loss", 0.1, patience=30, restore_best_weights=True, best_weight_identifier="val_loss")

start_cpu = time.process_time()

mae = tf.keras.metrics.MeanAbsoluteError(name="mae")
mae_rare = RareMAE(threshold=threshold, name="mae_rare")
aore = AORE(threshold=threshold, name="aore")
fp = RegressionFalsePositives(threshold=threshold, name="fp")
fn = RegressionFalseNegatives(threshold=threshold, name="fn")
f1 = RegressionF1Score(threshold=threshold, name="f1")

stage_one_model.compile(loss="mse",
              optimizer="adam",
              )

# --- modular CV setup ---
stage_one_strategy = DecoupledFitStageOneStrategy()

stage_one_params = DecoupledFitStageOneParams(
    x=x_train,
    y=y_train,
    sample_weight=sample_weights,
    # class_weight={
    #     0: common_weight,   # common class
    #     1: rare_weight,     # rare class
    # },
    batch_size=batch_size,
    shuffle=True,                # keep your intended behavior
    stratify_batches=True,        # matches your current call
    callbacks=[stage_one_convergence_stop],       # NOTE: only pass the EarlyStopping here if you want it cloned per fold
    kwargs={
        # anything else you used to pass through **kwargs to balanced_fit / fit
        # If your balanced_fit supports extra flags like generate_decoder_branch, put them here:
        # "generate_decoder_branch": True,
        # "representation_layer_index": -2,
    },
)

stage_one_history, stage_two_model, stage_one_selection_info = fit_k_folds_modular(
    model=stage_one_model,
    x=x_train,
    y=y_train,
    strategy=stage_one_strategy,
    params=stage_one_params,
    batch_size=batch_size,                 # driver-level override (kept consistent)
    num_folds=num_folds_for_split,
    shuffle=True,
    seed=seed,
    metric=monitor_metric,
    min_or_max="min",
    selection_metric="aore",
    selection_min_or_max="min",
    mode=ModelType.REGRESSION,
    metrics=[mae, mae_rare, aore, fp, fn, f1],
)

stage_two_convergence_stop = ConvergenceStopping("loss", 0.1, patience=30, restore_best_weights=True, best_weight_identifier="val_loss")

stage_two_strategy = DecoupledFitStageTwoExtraLayerStrategy()

stage_two_params = DecoupledFitStageTwoParams(
    x=x_train,
    y=y_train,
    #sample_weight=sample_weights,
    # class_weight={
    #     0: common_weight,   # common class
    #     1: rare_weight,     # rare class
    # },
    batch_size=batch_size,
    shuffle=True,                # keep your intended behavior
    stratify_batches=True,        # matches your current call
    callbacks=[stage_two_convergence_stop],       # NOTE: only pass the EarlyStopping here if you want it cloned per fold
    kwargs={

    },
)

stage_two_history, stage_two_model, stage_two_selection_info = fit_k_folds_modular(
    model=stage_two_model,
    x=x_train,
    y=y_train,
    strategy=stage_two_strategy,
    params=stage_two_params,
    batch_size=batch_size,                 # driver-level override (kept consistent)
    num_folds=num_folds_for_split,
    shuffle=True,
    seed=seed,
    metric=monitor_metric,
    min_or_max="min",
    selection_metric="aore",
    selection_min_or_max="min",
    mode=ModelType.REGRESSION,
    alpha_candidates=alpha_values,
    metrics=[mae, mae_rare, aore, fp, fn, f1],
)

end_cpu = time.process_time()

results = stage_two_model.evaluate(x_test, y_test, verbose=0)

loss = results
print(f"\nCPU time: {end_cpu - start_cpu:.4f} sec")
print(f"loss: {loss:.4f}")

cv_results = []

cv_by_candidate_stage_two = stage_two_selection_info["cv_by_candidate"]

for cand_name, summary in cv_by_candidate_stage_two.items():
    alpha = summary["meta"]["alpha"]
    avg = summary["avg_metrics_at_best_epoch"]
    mae_val = summary["mean_best_metric_value"]

    cv_results.append({
        "alpha": alpha,
        "avg_best_epoch": summary["avg_best_epoch"],
        "val_mae": mae_val,
        "val_rare_mae": avg.get("mae_rare"),
        "val_aore": avg.get("aore"),
        "val_fp": avg.get("fp"),
        "val_fn": avg.get("fn"),
        "val_f1": avg.get("f1"),
    })

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.precision', 4)
cv_df = pd.DataFrame(cv_results)
print("\n=== CV Sweep Summary ===")
print(cv_df)
print()

# -------------- Confusion Matrix ---------------
def run_regression_plot(data, target, threshold_regplot, model_sep_regplot, x_test_regplot):
    y_true = data[target].to_numpy()
    y_pred = model_sep_regplot.predict(x_test_regplot, batch_size=512, verbose=0).reshape(-1)

    plot_predicted_vs_actual(
        y_true=y_true,
        y_pred=y_pred,
        threshold=threshold_regplot,
        out_png="pred_vs_actual_ln_peak_decoupled_fit_extra_layer_current_model.png",
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

run_regression_plot(test_data, target_column, threshold, stage_two_model, x_test)

# ----------------------------------
# Summary table
# ----------------------------------
y_pred = stage_two_model.predict(x_test, verbose=0)

metrics_tuple = compute_regression_metrics(
    y_true=y_test,
    y_pred=y_pred,
    threshold=np.log(10.0),
)
overall_mae, rare_mae, _, aore, overall_pcc, rare_pcc, aorc, fp, fn, f1 = metrics_tuple

all_results.append({
    "alpha": stage_two_selection_info["selected_alpha"],
    "loss": loss,
    "overall_mae": overall_mae,
    "rare_mae": rare_mae,
    "aore": aore,
    #"overall_pcc": overall_pcc,
    #"rare_pcc": rare_pcc,
    #"aorc": aorc,
    "FP": fp,
    "FN": fn,
    "F1": f1,
    "epochs": len(stage_two_history.history["loss"]),
})

results_df = pd.DataFrame(all_results)
print("\n=== Best CV full run Summary ===")
print(results_df)

# from f1_threshold_sweep import f1_threshold_sweep
# test_threshold, test_f1, _, _ = f1_threshold_sweep(y_test_prob, y_test_true, 0.05)

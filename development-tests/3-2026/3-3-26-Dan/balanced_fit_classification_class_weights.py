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
from plot_auroc_curve import plot_auroc_curve
from custom_callbacks import ConvergenceStopping


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
monitor_metric = "val_f1_score"


# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_training.csv")
test_data  = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_testing.csv")

y_train = (train_data[target_column].values >= threshold).reshape(-1, 1).astype("float32")
y_test  = (test_data[target_column].values  >= threshold).reshape(-1, 1).astype("float32")

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test  = test_data.drop(columns=[target_column]).values.astype(np.float32)


# ----------------------------
# Model
# ----------------------------
def build_model(input_shape: int) -> imbal.classification.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
    hidden2 = layers.Dense(12, activation="relu", name="hidden_layer2")(hidden1)
    hidden3 = layers.Dense(8, activation="relu", name="hidden_layer3")(hidden2)
    hidden4 = layers.Dense(6, activation="relu", name="hidden_layer4")(hidden3)
    outputs = layers.Dense(1, activation="sigmoid", name="output_layer")(hidden4)
    built_model = imbal.classification.Model(inputs=inputs, outputs=outputs, name="one_hidden_layer_6_units")
    return built_model


def build_model_from_research_paper(input_shape: int) -> imbal.classification.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
    hidden2 = layers.Dense(9, activation="relu", name="hidden_layer2")(hidden1)
    hidden3 = layers.Dense(6, activation="relu", name="hidden_layer3")(hidden2)
    outputs = layers.Dense(1, activation="sigmoid", name="output_layer")(hidden3)
    built_model = imbal.classification.Model(inputs=inputs, outputs=outputs, name="one_hidden_layer_6_units")
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


class_weight_candidates=[
        {0: 0.9, 1: 0.1},
        {0: 0.8, 1: 0.2},
        {0: 0.7, 1: 0.3},
        {0: 0.6, 1: 0.4},
        {0: 0.5, 1: 0.5},
        {0: 0.4, 1: 0.6},
        {0: 0.3, 1: 0.7},
        {0: 0.2, 1: 0.8},
        {0: 0.1, 1: 0.9},
    ]
all_results = []

reset_seeds(seed)

model = build_model(x_train.shape[1])
#model = build_model_from_research_paper(x_train.shape[1])

#sample_weights = imbal.classification.generate_sample_weights(y_train)

# early_stop = keras.callbacks.EarlyStopping(
#     monitor="val_loss",
#     min_delta=early_stop_min_delta,
#     patience=early_stop_patience,
#     mode="max",
#     restore_best_weights=True,
#     verbose=1,
# )

convergence_stop = ConvergenceStopping("val_loss", 0.1, patience=30, restore_best_weights=True)

start_cpu = time.process_time()

f1 = tf.keras.metrics.F1Score(threshold=0.9)
auroc = tf.keras.metrics.AUC(curve="ROC", name="auroc")
fp = tf.keras.metrics.FalsePositives(thresholds=0.5, name="fp")
fn = tf.keras.metrics.FalseNegatives(thresholds=0.5, name="fn")

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=[f1, auroc, fp, fn],
              )

# --- modular CV setup ---
strategy = BalancedFitStrategy()

params = BalancedFitParams(
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
    callbacks=[convergence_stop],       # NOTE: only pass the EarlyStopping here if you want it cloned per fold
    kwargs={
        # anything else you used to pass through **kwargs to balanced_fit / fit
        # If your balanced_fit supports extra flags like generate_decoder_branch, put them here:
        # "generate_decoder_branch": True,
        # "representation_layer_index": -2,
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
    mode=ModelType.CLASSIFICATION,
    min_or_max="max",
    class_weight_candidates=class_weight_candidates,
)


end_cpu = time.process_time()

results = model.evaluate(x_test, y_test, verbose=0)

# -------------- Confusion Matrix ---------------
from f1_threshold_sweep import f1_threshold_sweep

best_threshold, best_f1score, best_counts, _ = f1_threshold_sweep(model.predict(x_test), y_test, 0.1, verbose=0)

loss, f1_val, auroc_val, _, _ = results
print(f"\nCPU time: {end_cpu - start_cpu:.4f} sec")
print(f"loss: {loss:.4f}")
print(f"f1_score: {f1_val:.4f}")
print(f"auroc: {auroc_val:.4f}")

cv_results = []

cv_by_candidate = selection_info["cv_by_candidate"]

for cand_name, summary in cv_by_candidate.items():
    cw = summary["meta"]["class_weight"]
    avg = summary["avg_metrics_at_best_epoch"]

    cv_results.append({
        "class_weights": cw,
        "avg_best_epoch": summary["avg_best_epoch"],
        "val_f1_score": avg.get("val_f1_score"),
        "val_auroc": avg.get("val_auroc"),
        "val_fp": avg.get("val_fp"),
        "val_fn": avg.get("val_fn"),
    })

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.precision', 4)
cv_df = pd.DataFrame(cv_results)
print("\n=== CV Sweep Summary ===")
print(cv_df)
print()

all_results.append({
    "class_weights": selection_info["selected_class_weight"],
    "loss": loss,
    "mean_best_metric_value": selection_info["mean_best_metric_value"],
    "threshold": best_threshold,
    "best_f1_score": best_f1score,
    "TP|FP|TN|FN": f"{best_counts["TP"]}|{best_counts["FP"]}|{best_counts["TN"]}|{best_counts["FN"]}",
    "auroc": auroc_val,
    "epochs": len(history.history["loss"]),
})

# ----------------------------------
# Summary table
# ----------------------------------
results_df = pd.DataFrame(all_results)
print("\n=== Best CV full run Summary ===")
print(results_df)

plot_auroc_curve(
   model,
   x_test,
   y_test,
   title="Balanced Fit (Current Model) Test ROC (colored by threshold)",
   color_by_threshold=True,
   save_path="balanced_fit_current_model_roc_curve.png",
)
# plot_auroc_curve(
#     model,
#     x_test,
#     y_test,
#     title="Balanced Fit (Paper Model) Test ROC (colored by threshold)",
#     color_by_threshold=True,
#     save_path="balanced_fit_paper_model_roc_curve.png",
# )

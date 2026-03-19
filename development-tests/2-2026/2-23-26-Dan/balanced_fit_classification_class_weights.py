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

weight_values = np.arange(0.1, 1.0, 0.1)
all_results = []

for common_weight in weight_values:
    reset_seeds(seed)

    rare_weight = round(1.0 - float(common_weight), 1)

    print("\n" + "=" * 60)
    print(f"Training with class_weight = {{0: {common_weight}, 1: {rare_weight}}}")
    print("=" * 60)

    model = build_model(x_train.shape[1])
    #model = build_model_from_research_paper(x_train.shape[1])

    #sample_weights = imbal.classification.generate_sample_weights(y_train)

    early_stop = keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        min_delta=early_stop_min_delta,
        patience=early_stop_patience,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    )

    start_cpu = time.process_time()

    f1 = tf.keras.metrics.F1Score(threshold=0.5)
    auroc = tf.keras.metrics.AUC(curve="ROC", name="auroc")

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=[f1, auroc],
                  )

    # --- modular CV setup ---
    strategy = BalancedFitStrategy()

    params = BalancedFitParams(
        x=x_train,
        y=y_train,
        #sample_weight=sample_weights,
        class_weight={
            0: common_weight,   # common class
            1: rare_weight,     # rare class
        },
        batch_size=batch_size,
        shuffle=True,                # keep your intended behavior
        stratify_batches=True,        # matches your current call
        callbacks=[early_stop],       # NOTE: only pass the EarlyStopping here if you want it cloned per fold
        kwargs={
            # anything else you used to pass through **kwargs to balanced_fit / fit
            # If your balanced_fit supports extra flags like generate_decoder_branch, put them here:
            # "generate_decoder_branch": True,
            # "representation_layer_index": -2,
        },
    )

    history = fit_k_folds_modular(
        model=model,
        x=x_train,
        y=y_train,
        strategy=strategy,
        params=params,
        batch_size=batch_size,                 # driver-level override (kept consistent)
        num_folds=num_folds_for_split,
        shuffle=True,
        seed=seed,
        mode=ModelType.CLASSIFICATION,
    )


    end_cpu = time.process_time()

    results = model.evaluate(x_test, y_test, verbose=0)

    # -------------- Confusion Matrix ---------------
    from f1_threshold_sweep import f1_threshold_sweep

    best_threshold, best_f1score, best_counts, _ = f1_threshold_sweep(model.predict(x_test), y_test, 0.1, verbose=0)

    loss, f1_val, auroc_val = results
    print(f"\nCPU time: {end_cpu - start_cpu:.4f} sec")
    print(f"loss: {loss:.4f}")
    print(f"f1_score: {f1_val:.4f}")
    print(f"auroc: {auroc_val:.4f}")

    all_results.append({
        "common_weight": common_weight,
        "rare_weight": rare_weight,
        "loss": loss,
        "threshold": best_threshold,
        "best_f1_score": best_f1score,
        "TP|FP|TN|FN": f"{best_counts["TP"]}|{best_counts["FP"]}|{best_counts["TN"]}|{best_counts["FN"]}",
        "auroc": auroc_val,
        "epochs": len(history.history["loss"]),
    })

# plot_auroc_curve(
#    model,
#    x_test,
#    y_test,
#    title="Balanced Fit (Current Model) Test ROC (colored by threshold)",
#    color_by_threshold=True,
#    save_path="balanced_fit_current_model_roc_curve.png",
# )
# plot_auroc_curve(
#     model,
#     x_test,
#     y_test,
#     title="Balanced Fit (Paper Model) Test ROC (colored by threshold)",
#     color_by_threshold=True,
#     save_path="balanced_fit_paper_model_roc_curve.png",
# )

# ----------------------------------
# Summary table
# ----------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
results_df = pd.DataFrame(all_results)
print("\n=== Sweep Summary ===")
print(results_df)

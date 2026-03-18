import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import time
import tensorflow as tf
import imbal

from sklearn.model_selection import StratifiedKFold

# ----------------------------
# Config
# ----------------------------
seed = 42
tf.keras.utils.set_random_seed(
    seed
)

target_column = "ln_peak_intensity"
threshold = np.log(10.0)

num_folds = 5
max_epochs = 300
batch_size = 512
seed = 42

# Local-minimum "patience" rule:
# consider epoch i a local min if loss improved at i vs i-1,
# and then fails to improve for `patience` epochs after i.
local_min_patience = 30
local_min_min_delta = 0.0  # set to e.g. 1e-4 if you want strict improvements

# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_training.csv")
test_data  = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_testing.csv")

y_train = (train_data[target_column].values >= threshold).astype(int).reshape(-1, 1).astype("float32")
y_test  = (test_data[target_column].values  >= threshold).astype(int).reshape(-1, 1).astype("float32")

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test  = test_data.drop(columns=[target_column]).values.astype(np.float32)

# ----------------------------
# Model
# ----------------------------
def build_model(input_shape: int) -> keras.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
    hidden2 = layers.Dense(12, activation="relu", name="hidden_layer2")(hidden1)
    hidden3 = layers.Dense(8,  activation="relu", name="hidden_layer3")(hidden2)
    hidden4 = layers.Dense(6,  activation="relu", name="hidden_layer4")(hidden3)
    outputs = layers.Dense(1, activation="sigmoid", name="output_layer")(hidden4)
    return keras.Model(inputs=inputs, outputs=outputs, name="sep_model")

def make_compile_params():
    f1 = tf.keras.metrics.F1Score(threshold=0.5)
    return imbal.classification.wrap_model_compile_parameters(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=[f1],
    )

# ----------------------------
# Epoch selection from val_loss
# ----------------------------
def pick_epoch_from_val_loss(val_loss, patience=10, min_delta=0.0):
    """
    Returns a 1-based epoch index.

    Rule:
      - Find the first "local minimum" where val_loss improves vs previous epoch,
        and then does NOT improve (by min_delta) for `patience` subsequent epochs.
      - If none found, fall back to the global minimum epoch.
    """
    v = np.asarray(val_loss, dtype=float)

    def improved(a, b):
        # True if b is lower than a by at least min_delta
        return (a - b) > min_delta

    # scan for a local minimum according to the patience rule
    for i in range(1, len(v) - patience):
        # candidate minimum at i if it improved vs i-1
        if improved(v[i - 1], v[i]):
            # check if no further improvements for `patience` epochs
            window = v[i + 1 : i + 1 + patience]
            if all(not improved(v[i], w) for w in window):
                return i + 1  # 1-based

    # fallback: global min
    return int(np.argmin(v) + 1)

print("Starting K-fold cross-validation...")

fold_metrics = []
ideal_epochs = []

# StratifiedKFold wants 1D labels
y_train_1d = y_train.reshape(-1).astype(int)

kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

start_cpu = time.process_time()

fold_no = 1
for tr_idx, va_idx in kfold.split(x_train, y_train_1d):

    print("------------------------------------------------------------------------")
    print(f"Training for fold {fold_no} ...")

    X_tr, X_va = x_train[tr_idx], x_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    # Fresh model per fold
    model = build_model(x_train.shape[1])

    # Sample weights per fold
    sample_weights = imbal.classification.generate_sample_weights(y_tr)

    # Train, monitoring val_loss
    history = imbal.classification.balanced_fit(
        model,
        X_tr,
        y_tr,
        sample_weights=sample_weights,
        compile_parameters=make_compile_params(),
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(X_va, y_va),
    )

    # history is a keras.callbacks.History object
    hist = history.history.history  # <-- dict
    print(hist)

    print("History keys:", hist.keys())  # e.g. dict_keys(['loss','f1_score','val_loss','val_f1_score'])
    print("First few losses:", hist["loss"][:3])
    print("First few val_losses:", hist.get("val_loss", [])[:3])

    chosen_epoch = pick_epoch_from_val_loss(
        hist["val_loss"],
        patience=local_min_patience,
        min_delta=local_min_min_delta,
    )
    ideal_epochs.append(chosen_epoch)

    # Evaluate fold
    scores = model.evaluate(X_va, y_va, verbose=0)
    metric_dict = dict(zip(model.metrics_names, scores))
    fold_metrics.append(metric_dict)

    # Print fold summary
    best_val = float(np.min(np.asarray(hist["val_loss"], dtype=float)))
    print(f"Fold {fold_no}: chosen ideal epoch = {chosen_epoch}")
    print(f"Fold {fold_no}: best val_loss observed = {best_val:.6f}")
    print("Fold metrics:", {k: float(v) for k, v in metric_dict.items()})

    fold_no += 1

end_cpu = time.process_time()
print(f"\nCPU time spent (CV): {end_cpu - start_cpu:.4f} seconds")

# ----------------------------
# Summarize ideal epoch from CV
# ----------------------------
ideal_epochs = np.asarray(ideal_epochs, dtype=int)
ideal_epoch_mean = float(np.mean(ideal_epochs))
ideal_epoch_std  = float(np.std(ideal_epochs, ddof=1)) if len(ideal_epochs) > 1 else 0.0
ideal_epoch_final = int(np.round(ideal_epoch_mean))

print("\n------------------------------------------------------------------------")
print("Ideal epoch (per fold):", ideal_epochs.tolist())
print(f"Ideal epoch mean = {ideal_epoch_mean:.2f} (+/- {ideal_epoch_std:.2f})")
print(f"Chosen ideal epoch for rerun (rounded mean): {ideal_epoch_final}")
print("------------------------------------------------------------------------")

# ----------------------------
# Rerun: train on full training data with chosen epoch count
# ----------------------------
print("\nRetraining on full training set with chosen ideal epoch count...")

model_sep = build_model(x_train.shape[1])
full_sample_weights = imbal.classification.generate_sample_weights(y_train)

start_cpu = time.process_time()
imbal.classification.balanced_fit(
    model_sep,
    x_train,
    y_train,
    sample_weights=full_sample_weights,
    compile_parameters=make_compile_params(),
    epochs=ideal_epoch_final,
    batch_size=batch_size
)
end_cpu = time.process_time()
print(f"CPU time spent (final train): {end_cpu - start_cpu:.4f} seconds")

# ----------------------------
# Final holdout evaluation
# ----------------------------
results = model_sep.evaluate(x_test, y_test, verbose=0)

print("\n=== Model test results (holdout test set) ===")
for name, value in zip(model_sep.metrics_names, results):
    print(f"{name}: {value:.4f}")

# ----------------------------
# Summarize ideal epoch from CV
# ----------------------------
ideal_epochs = np.asarray(ideal_epochs, dtype=int)
ideal_epoch_mean = float(np.mean(ideal_epochs))
ideal_epoch_std  = float(np.std(ideal_epochs, ddof=1)) if len(ideal_epochs) > 1 else 0.0
ideal_epoch_final = int(np.round(ideal_epoch_mean))

print("\n------------------------------------------------------------------------")
print("Ideal epoch (per fold):", ideal_epochs.tolist())
print(f"Ideal epoch mean = {ideal_epoch_mean:.2f} (+/- {ideal_epoch_std:.2f})")
print(f"Chosen ideal epoch for rerun (rounded mean): {ideal_epoch_final}")
print("------------------------------------------------------------------------")
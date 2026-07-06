import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import layers
import os
from aore_metric import AORE

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
train_data = pd.read_csv("../../../../tutorials/data/SEP-C/sep_10mev_training.csv")
test_data  = pd.read_csv("../../../../tutorials/data/SEP-C/sep_10mev_testing.csv")

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
loss, _, mae = results
predictions = model.predict(x_test)

print(f"Test Loss: {loss:.4f}")
print(f"Test MAE: {mae:.4f}")

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

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import layers

import imbal

seed = 42
tf.keras.utils.set_random_seed(
    seed
)

target_column = "ln_peak_intensity"

max_epochs = 300
batch_size = 32

# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv("sep_model_training_regression.csv")
test_data  = pd.read_csv("sep_model_testing_regression.csv")

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
    built_model = imbal.regression.Model(inputs=inputs, outputs=outputs, name="sep_model")
    return built_model

model = build_model(x_train.shape[1])


# ----------------------------
# Training
# ----------------------------
labels_kde = y_train.reshape(-1).copy()
kde = imbal.regression.fit_kde(labels_kde)
densities = imbal.regression.get_sample_densities(labels_kde, kde)

model.compile(loss="mean_squared_error",
              optimizer="adam",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
              )

from imbal.regression import reciprocal_importance
weights = reciprocal_importance(densities, alpha=0.8)
model.balanced_fit(x_train,
                   y_train,
                   sample_weight=weights,
                   batch_size=batch_size,
                   epochs=max_epochs,
                   )


# ----------------------------
# Evaluation
# ----------------------------
rare_threshold = np.log(10.0)

common_mask = y_test.reshape(-1) < rare_threshold
rare_mask = y_test.reshape(-1) >= rare_threshold

y_pred = model.predict(x_test)

results = model.evaluate(x_test, y_test)
loss, mae = results

mae_common_metric = keras.metrics.MeanAbsoluteError(name="common_mae")
mae_common_metric.update_state(y_test[common_mask], y_pred[common_mask])
mae_common = mae_common_metric.result()

mae_rare_metric = keras.metrics.MeanAbsoluteError(name="rare_mae")
mae_rare_metric.update_state(y_test[rare_mask], y_pred[rare_mask])
mae_rare = mae_rare_metric.result()

mse_metric = keras.metrics.MeanSquaredError(name="mse")
mse_metric.update_state(y_test, y_pred)
mse = mse_metric.result()

mse_common_metric = keras.metrics.MeanSquaredError(name="common_mse")
mse_common_metric.update_state(y_test[common_mask], y_pred[common_mask])
mse_common = mse_common_metric.result()

mse_rare_metric = keras.metrics.MeanSquaredError(name="rare_mse")
mse_rare_metric.update_state(y_test[rare_mask], y_pred[rare_mask])
mse_rare = mse_rare_metric.result()

pcc_metric = keras.metrics.PearsonCorrelation(name="pcc", axis=0)
pcc_metric.update_state(y_test, y_pred)
pcc = pcc_metric.result().numpy()

pcc_common_metric = keras.metrics.PearsonCorrelation(name="common_pcc", axis=0)
pcc_common_metric.update_state(y_test[common_mask], y_pred[common_mask])
pcc_common = pcc_common_metric.result().numpy()

pcc_rare_metric = keras.metrics.PearsonCorrelation(name="rare_pcc", axis=0)
pcc_rare_metric.update_state(y_test[rare_mask], y_pred[rare_mask])
pcc_rare = pcc_rare_metric.result().numpy()

print(f"Test Loss: {loss:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test Common Sample MAE: {mae_common:.4f}")
print(f"Test Rare Sample MAE: {mae_rare:.4f}")
print(f"Test MSE: {mse:.4f}")
print(f"Test Common Sample MSE: {mse_common:.4f}")
print(f"Test Rare Sample MSE: {mse_rare:.4f}")
print(f"Test PCC: {pcc:.4f}")
print(f"Test Common Sample PCC: {pcc_common:.4f}")
print(f"Test Rare Sample PCC: {pcc_rare:.4f}")


# ----------------------------
# Visualization
# ----------------------------
imbal.regression.plot_true_vs_predictions(y_test, y_pred)

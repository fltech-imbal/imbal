import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import time
import tensorflow as tf
import imbal

seed = 42
tf.keras.utils.set_random_seed(
    seed
)

target_column = "ln_peak_intensity"
threshold = np.log(10.0)

train_data = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_training.csv")
test_data = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_testing.csv")

y_train = (train_data[target_column].values >= threshold)
y_test = (test_data[target_column].values >= threshold)

y_train = y_train.reshape(-1, 1).astype("float32")
y_test = y_test.reshape(-1, 1).astype("float32")

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test = test_data.drop(columns=[target_column]).values.astype(np.float32)

sample_weights = imbal.classification.generate_sample_weights(y_train)


def build_model(input_shape: int) -> imbal.classification.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
    hidden2 = layers.Dense(12, activation="relu", name="hidden_layer2")(hidden1)
    hidden3 = layers.Dense(8, activation="relu", name="hidden_layer3")(hidden2)
    hidden4 = layers.Dense(6, activation="relu", name="hidden_layer4")(hidden3)
    outputs = layers.Dense(1, activation="sigmoid", name="output_layer")(hidden4)
    model = imbal.classification.Model(inputs=inputs, outputs=outputs, name="one_hidden_layer_6_units")
    return model


model_sep = build_model(x_train.shape[1])

f1 = tf.keras.metrics.F1Score(threshold=0.5)
auroc = tf.keras.metrics.AUC(curve="ROC", name="auroc")

model_sep.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=[f1, auroc],
                  stratify_batches=True,
                  )

start_cpu = time.process_time()

model_sep.cRT_fit(
    x_train,
    y_train,
    sample_weight=sample_weights,
    epochs=100,
    batch_size=512,
)

end_cpu = time.process_time()

cpu_time_seconds = end_cpu - start_cpu
print(f"CPU time spent: {cpu_time_seconds:.4f} seconds")

results = model_sep.evaluate(x_test, y_test, verbose=0)

loss, f1_val, auroc_val = results
print("\n=== Model test results ===")
print(f"loss: {loss:.4f}")
print(f"f1_score: {f1_val:.4f}")
print(f"auroc: {auroc_val:.4f}")


# -------------- Confusion Matrixx ---------------
y_prob = model_sep.predict(x_test, batch_size=256, verbose=0)
y_pred = (y_prob >= 0.5).astype(int)
y_true = y_test.astype(int)

TP = np.sum((y_true == 1) & (y_pred == 1))
TN = np.sum((y_true == 0) & (y_pred == 0))
FP = np.sum((y_true == 0) & (y_pred == 1))
FN = np.sum((y_true == 1) & (y_pred == 0))

print("\n=== Confusion Matrix ===")
print(f"TP: {TP}")
print(f"FP: {FP}")
print(f"TN: {TN}")
print(f"FN: {FN}")

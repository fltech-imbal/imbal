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
train_data = pd.read_csv("sep_model_training_classification.csv")
test_data  = pd.read_csv("sep_model_testing_classification.csv")

y_train = train_data[target_column].values.reshape(-1, 1).astype("float32")
y_test  = test_data[target_column].values.reshape(-1, 1).astype("float32")

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
    built_model = imbal.classification.Model(inputs=inputs, outputs=outputs, name="sep_model")
    return built_model

model = build_model(x_train.shape[1])


# ----------------------------
# Training
# ----------------------------
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=[imbal.metrics.HeikdeSkillScore(threshold=0.5, name="HSS")],
              )

class_weights = {0: 0.9, 1: 0.1}

model.balanced_fit(x_train,
                   y_train,
                   class_weight=class_weights,
                   batch_size=batch_size,
                   epochs=max_epochs,
                   )


# ----------------------------
# Evaluation
# ----------------------------
y_pred = model.predict(x_test)

results = model.evaluate(x_test, y_test)
loss, hss = results

tss_metric = imbal.metrics.TrueSkillStatistic(threshold=0.5)
tss_metric.update_state(y_test, y_pred)

auc_metric = imbal.metrics.BoundedAUC(num_thresholds=50)
auc_metric.update_state(y_test, y_pred)

f1_metric = keras.metrics.F1Score(threshold=0.5)
f1_metric.update_state(y_test, y_pred)

j_stat_metric = imbal.metrics.JStatistic(threshold=0.5)
j_stat_metric.update_state(y_test, y_pred)

youdens_index_metric = imbal.metrics.YoudensIndex(threshold=0.5)
youdens_index_metric.update_state(y_test, y_pred)

gilbert_skill_score_metric = imbal.metrics.GilbertSkillScore(threshold=0.5)
gilbert_skill_score_metric.update_state(y_test, y_pred)

critical_success_index_metric = imbal.metrics.CriticalSuccessIndex(threshold=0.5)
critical_success_index_metric.update_state(y_test, y_pred)

print(f"Test Loss: {loss:.4f}")
print(f"Test HSS: {hss:.4f}")
print(f"Test TSS: {tss_metric.result().numpy().item():.4f}")
print(f"Test AUC: {auc_metric.result():.4f}")
print(f"Test F1Score: {f1_metric.result().numpy().item():.4f}")
print(f"Test J Statistic: {j_stat_metric.result().numpy().item():.4f}")
print(f"Test Youden's Index: {youdens_index_metric.result().numpy().item():.4f}")
print(f"Test Gilbert Skill Score: {gilbert_skill_score_metric.result().numpy().item():.4f}")
print(f"Test Critical Success Index: {critical_success_index_metric.result().numpy().item():.4f}")


# ----------------------------
# Visualization
# ----------------------------
imbal.classification.plot_confusion_matrix(y_test, y_pred)

imbal.classification.plot_roc(y_test, y_pred)
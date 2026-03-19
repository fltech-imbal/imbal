import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from sep_regression_plot import plot_predicted_vs_actual

seed = 42
tf.keras.utils.set_random_seed(
    seed
)

target_column = "ln_peak_intensity"
threshold = np.log(10.0)

train_data = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_training.csv")
test_data = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_testing.csv")

y_train = train_data[target_column].values.astype(np.float32).reshape(-1, 1)
y_test  = test_data[target_column].values.astype(np.float32).reshape(-1, 1)

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test = test_data.drop(columns=[target_column]).values.astype(np.float32)


def build_model(input_shape: int) -> keras.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
    hidden2 = layers.Dense(12, activation="relu", name="hidden_layer2")(hidden1)
    hidden3 = layers.Dense(8, activation="relu", name="hidden_layer3")(hidden2)
    hidden4 = layers.Dense(6, activation="relu", name="hidden_layer4")(hidden3)
    outputs = layers.Dense(1, activation="linear", name="output_layer")(hidden4)
    model = keras.Model(inputs=inputs, outputs=outputs, name="one_hidden_layer_6_units")
    return model


model = build_model(x_train.shape[1])

start_cpu = time.process_time()

model.compile(
    optimizer="adam",
    loss="mae",
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae")
    ]
)


early_stop = keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=0.001,
    patience=50,
    mode="min",
    restore_best_weights=True,
    verbose=1,
)

history = model.fit(x_train, y_train,
                          epochs=1000,
                          batch_size=512,
                          callbacks=[early_stop])

end_cpu = time.process_time()

cpu_time_seconds = end_cpu - start_cpu
print(f"CPU time spent: {cpu_time_seconds:.4f} seconds")

results = model.evaluate(x_test, y_test, verbose=0)

print("\n=== Model test results ===")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")


# -------------- Regression Plot ---------------
def run_regression_plot(data, target, threshold_regplot, model_sep_regplot, x_test_regplot):
    y_true = data[target].to_numpy()
    y_pred = model_sep_regplot.predict(x_test_regplot, batch_size=512, verbose=0).reshape(-1)

    plot_predicted_vs_actual(
        y_true=y_true,
        y_pred=y_pred,
        threshold=threshold_regplot,
        out_png="pred_vs_actual_ln_peak_regular_fit.png",
        title="Predicted vs Actual ln(peak intensity)",
        show=False
    )


run_regression_plot(test_data, target_column, threshold, model, x_test)

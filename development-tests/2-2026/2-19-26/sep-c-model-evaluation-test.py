import keras
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math, time, imbal
from sklearn.metrics import roc_curve, auc
from matplotlib.collections import LineCollection

MODEL_TASK = 'classification'

MODE = 'balanced'
STRATIFY = True
AE = False
REPRESENTATION_LAYER_INDEX = -2
GEN_OUTPUT = True
batch_size = 512
epochs = 10000
LEARNING_RATE =4e-4


"""
Load data
"""
seed = 42
tf.keras.utils.set_random_seed(
    seed
)

target_column = "ln_peak_intensity"

train_data = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_training.csv")
test_data = pd.read_csv("../../../tutorials/data/SEP-C/sep_10mev_testing.csv")

y_train = train_data[target_column].values
y_test = test_data[target_column].values

y_train = y_train.reshape(-1).astype("float32")
y_test = y_test.reshape(-1).astype("float32")

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test = test_data.drop(columns=[target_column]).values.astype(np.float32)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

scaler = StandardScaler()
x_combined = np.concatenate((x_train, x_test), axis=0)
y_combined = np.concatenate((y_train, y_test), axis=0)
scaled_x_combined = scaler.fit_transform(x_combined)

x_train = scaled_x_combined[:x_train.shape[0]]
x_test = scaled_x_combined[x_train.shape[0]:]

if MODEL_TASK == 'classification':
    y_train = y_train >= math.log(10)
    y_test = y_test >= math.log(10)
print('x_train', x_train.shape)
print('y_train',y_train.shape)
print('x_test',x_test.shape)
print('y_test',y_test.shape)
print(np.min(y_train), np.max(y_train))

print(y_train.shape)
print(x_test.shape)


"""
Build model
"""
input_shape = x_train.shape[1:]

inputs = keras.Input(shape=input_shape)
x = layers.Dense(18, activation='relu')(inputs)
x = layers.Dense(9, activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(6, activation='relu')(x)
x = layers.Flatten()(x)
output = layers.Dense(1, activation='sigmoid' if MODEL_TASK == 'classification' else 'linear')(x)

model = (
    imbal.classification.Model(inputs=inputs, outputs=output)
    if MODEL_TASK == 'classification'
    else imbal.regression.Model(inputs=inputs, outputs=output)
)

model.summary()

model.compile(
    loss="binary_crossentropy" if MODEL_TASK == 'classification' else 'mse',
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["accuracy" if MODEL_TASK == 'classification' else "mse"],
    generate_decoder_branch=AE,
    representation_layer_index=REPRESENTATION_LAYER_INDEX
)

print(model.get_compile_config())

subpackage = imbal.classification if MODEL_TASK == 'classification' else imbal.regression

"""
Generate sample densities
"""
BIN_COUNT=64
kde_bandwidth = imbal.regression.fit_kde(
    y_train,
    bin_count=BIN_COUNT
)
densities = imbal.regression.get_sample_densities(
    y_train,
    kde_bandwidth,
)

"""
Train model
"""
fit_function = model.fit
if MODE == 'balanced':
    fit_function = model.balanced_fit
if MODE == 'decoupled':
    fit_function = model.decoupled_fit

start = time.time()
weights = np.ones(x_train.shape[0])
if MODE == 'decoupled' and MODEL_TASK == 'classification':
    densities = imbal.regression.get_sample_densities(
        y_train,
        kde_bandwidth
    )
    weights = imbal.regression.generate_sample_weights(densities)

elif MODE == 'balanced' and MODEL_TASK == 'classification':
    densities = imbal.regression.get_sample_densities(
        y_train,
        kde_bandwidth
    )
    weights = imbal.regression.generate_sample_weights(densities)

model.override_second_stage_fit_parameters(
    epochs=epochs,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    ]
)

print(fit_function)
history = fit_function(
    x_train,
    y_train,
    stratify_batches=STRATIFY,
    validation_split=0.2,
    sample_weight=None if MODEL_TASK == 'classification' else weights   ,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    ]
)

if (MODE == 'decoupled'):
    one, two = history
    print('Stage lengths:')
    print(len(one.epoch), len(two.epoch))
end = time.time()

"""
Evaluate model
"""
print('EXECUTION TIME:', end - start)
print('Evaluating model...')
model.evaluate(x_test, y_test)
predictions = model.predict(x_test)

imbal.regression.plot_kde_1d(
    y_train,
    kde_bandwidth,
    bin_count=BIN_COUNT,
    save_figure='sep-c-data-distribution.png' if GEN_OUTPUT else None
)

if MODEL_TASK == 'classification':
    imbal.classification.tsne_visualization(
        model,
        x_test,
        y_test,
        representation_layer_index=REPRESENTATION_LAYER_INDEX,
        save_figure=f'tsne_visualization-{MODE}-ae-{AE}-rep{REPRESENTATION_LAYER_INDEX}.png' if GEN_OUTPUT else None,
    )
else:
    imbal.regression.tsne_visualization(
        model,
        x_test,
        y_test,
        representation_layer_index=REPRESENTATION_LAYER_INDEX,
        save_figure=f'tsne_visualization-{MODE}-ae-{AE}-rep{REPRESENTATION_LAYER_INDEX}.png' if GEN_OUTPUT else None,
    )

predictions = predictions.reshape(-1,)

y_test_labels = y_test
predictions_labels = (predictions >= 0.5).astype(int)

if MODEL_TASK == 'classification':
    cm = confusion_matrix(y_test_labels, predictions_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot()
    if GEN_OUTPUT:
        plt.savefig(f'confusion-matrix-{MODE}-ae-{AE}-rep{REPRESENTATION_LAYER_INDEX}.png')
    plt.show()

    f1_score = tf.keras.metrics.F1Score(threshold=0.5)
    f1_score.update_state(y_test_labels.reshape(-1, 1), predictions.reshape(-1, 1))

    auroc = tf.keras.metrics.AUC(num_thresholds=2000)
    print(y_test_labels.shape)
    print(y_test_labels[:20])
    print(predictions.shape)
    auroc.update_state(y_test_labels, predictions)
    print(auroc.result())

    print(np.max(predictions[y_test_labels == 0]))
    print(predictions[y_test_labels == 1][:20])

    y_scores = predictions

    fpr, tpr, thresholds = roc_curve(y_test_labels, y_scores, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    print("sklearn AUROC:", roc_auc)

    plt.figure(figsize=(7, 6))


    points = np.array([fpr, tpr]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm_thresholds = thresholds

    lc = LineCollection(
        segments,
        cmap='viridis',
        norm= plt.Normalize(vmin=0, vmax=1)
    )
    lc.set_array(norm_thresholds)
    lc.set_linewidth(2)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.add_collection(lc)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Decision Threshold")

    if GEN_OUTPUT:
        plt.savefig(f'roc-curve-{MODE}-ae-{AE}-rep{REPRESENTATION_LAYER_INDEX}.png')
    plt.show()

    print(f1_score.result())
else:
    y_rare_mask = y_test_labels > math.log(10)
    y_common_mask = y_test_labels <= math.log(10)
    plt.figure(figsize=(7, 6))
    plt.plot([-2.5, 8], [-2.5, 8], linestyle="--", linewidth=1, color='black', label="Perfect Prediction")
    plt.scatter(y_test_labels[y_common_mask], predictions.reshape(-1)[y_common_mask], color="red", alpha=0.3)
    plt.scatter(y_test_labels[y_rare_mask], predictions.reshape(-1)[y_rare_mask], color="#00ff00", alpha=0.3)
    plt.plot([-10, 10], [math.log(10), math.log(10)], color='#00000040', linestyle="--")
    plt.plot([math.log(10), math.log(10)], [-10, 10], color='#00000040', linestyle="--")
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.xlim(-2.5, 8.5)
    plt.ylim(-2.5, 8.5)
    if GEN_OUTPUT:
        plt.savefig(f'regression-true-pred-{MODE}-ae-{AE}-rep{REPRESENTATION_LAYER_INDEX}.png')
    plt.show()

    mask = y_test <= math.log(10)
    rare_mask = y_test > math.log(10)
    common_predictions = predictions[mask]
    rare_predictions = predictions[rare_mask]

    common_labels = y_test[mask]
    rare_labels = y_test[rare_mask]

    mse_common = np.mean(np.square(common_predictions - common_labels))
    mse_rare = np.mean(np.square(rare_predictions - rare_labels))

    print('common', f'{mse_common:.5f}')
    print('rare', f'{mse_rare:.5f}')

if MODE == 'decoupled':
    history = history[0]
print(history.history['val_loss'])





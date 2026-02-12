import keras
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os, math, time, imbal
import matplotlib.pyplot as plt

MODEL_TASK = 'regression'

MODE = 'decoupled'
STRATIFY = True
AE = True
REPRESENTATION_LAYER_INDEX = -4
GEN_OUTPUT = True

batch_size = 512
epochs = 8000
LEARNING_RATE =5e-4
STOPPING_PATIENCE=15

TRAIN_SPLIT = 0.8

PATH_START = '/mnt/c/Users/tommy/PycharmProjects/DrChanWorkPlayground'
print(os.getcwd())

cropped_folder = os.path.join(PATH_START, 'AgeDB/cropped')

print('loading data...')
y_data = np.load(os.path.join(cropped_folder, 'age_labels.npy')).reshape(-1)
x_data = np.load(os.path.join(cropped_folder, 'cropped_resized_images.npy'))
print('data loaded!')

print()
print(y_data.shape)
print(x_data.shape)
plt.imshow(x_data[0])
plt.show()

print('min', y_data.min())
print('max', y_data.max())

y_train = y_data[:round(len(y_data)*TRAIN_SPLIT)]
y_test = y_data[round(len(y_data)*TRAIN_SPLIT):]
x_train = x_data[:round(len(x_data)*TRAIN_SPLIT)]
x_test = x_data[round(len(x_data)*TRAIN_SPLIT):]

y_train = np.array(y_train).reshape(-1)
y_test = np.array(y_test).reshape(-1)

print(type(x_train))

print('train')
print(len(x_train))
print(y_train.shape)
print('test')
print(len(x_test))
print(y_test.shape)

inputs = keras.Input(shape=(112, 88, 3))

block_0_layer_0 = layers.Conv2D(16, 1, activation='relu', padding='same')(inputs)

block_1_layer_0 = layers.Conv2D(16, 3, activation='relu', padding='same')(block_0_layer_0)
block_1_layer_1 = layers.BatchNormalization()(block_1_layer_0)
block_1_layer_2 = layers.ReLU()(block_1_layer_1)
block_1_layer_3 = layers.Conv2D(16, 3, activation='relu', padding='same')(block_1_layer_2)
block_1_layer_4 = layers.BatchNormalization()(block_1_layer_3)
block_1_layer_5 = layers.Add()([block_0_layer_0, block_1_layer_4])
block_1_layer_6 = layers.ReLU()(block_1_layer_5)

block_2_layer_0 = layers.Conv2D(16, 3, padding='same')(block_1_layer_6)
block_2_layer_1 = layers.BatchNormalization()(block_2_layer_0)
block_2_layer_2 = layers.ReLU()(block_2_layer_1)
block_2_layer_3 = layers.Conv2D(16, 3, padding='same')(block_2_layer_2)
block_2_layer_4 = layers.BatchNormalization()(block_2_layer_3)
block_2_layer_5 = layers.Add()([block_1_layer_6, block_2_layer_4])
block_2_layer_6 = layers.ReLU()(block_2_layer_5)

block_3_layer_0 = layers.Conv2D(32, 3, strides=(2, 2), padding='same')(block_2_layer_6)
block_3_layer_1 = layers.BatchNormalization()(block_3_layer_0)
block_3_layer_2 = layers.ReLU()(block_3_layer_1)
block_3_layer_3 = layers.Conv2D(32, 3, padding='same')(block_3_layer_2)
block_3_layer_4 = layers.BatchNormalization()(block_3_layer_3)
block_3_shortcut_reduction = layers.Conv2D(32, 1, strides=(2, 2), padding='same')(block_2_layer_6)
block_3_layer_5 = layers.Add()([block_3_shortcut_reduction, block_3_layer_4])
block_3_layer_6 = layers.ReLU()(block_3_layer_5)

block_4_layer_0 = layers.Conv2D(32, 3, padding='same')(block_3_layer_6)
block_4_layer_1 = layers.BatchNormalization()(block_4_layer_0)
block_4_layer_2 = layers.ReLU()(block_4_layer_1)
block_4_layer_3 = layers.Conv2D(32, 3, padding='same')(block_4_layer_2)
block_4_layer_4 = layers.BatchNormalization()(block_4_layer_3)
block_4_layer_5 = layers.Add()([block_3_layer_6, block_4_layer_4])
block_4_layer_6 = layers.ReLU()(block_4_layer_5)

x = layers.Flatten()(block_4_layer_6)
x = layers.Dense(32, activation='relu')(x)
x = layers.Flatten()(x)
output = layers.Dense(1, activation='sigmoid' if MODEL_TASK == 'classification' else 'linear')(x)

model = (
    imbal.classification.Model(inputs=inputs, outputs=output)
    if MODEL_TASK == 'classification'
    else imbal.regression.Model(inputs=inputs, outputs=output)
)

model.summary()

print('number of layers', len(model.layers))

auc = keras.metrics.AUC(multi_label=True)

model.compile(
    loss="binary_crossentropy" if MODEL_TASK == 'classification' else 'mse',
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["accuracy" if MODEL_TASK == 'classification' else "mse"],
    generate_decoder_branch=AE,
    representation_layer_index=REPRESENTATION_LAYER_INDEX
)
BIN_COUNT=98

kde_bandwidth = imbal.regression.fit_kde(
    y_train,
    bin_count=BIN_COUNT
)
densities = imbal.regression.get_sample_densities(
    y_train,
    kde_bandwidth,
)

fit_function = model.fit
if MODE == 'balanced':
    fit_function = model.balanced_fit
if MODE == 'decoupled':
    fit_function = model.decoupled_fit

history = None

start = time.time()
weights = np.ones(len(x_train)).reshape(-1, 1)
if MODE == 'decoupled':
    densities = imbal.regression.get_sample_densities(
        y_train,
        kde_bandwidth
    )
    weights = imbal.regression.generate_sample_weights(densities)
    model.override_second_stage_fit_parameters(callbacks=[
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    ])
    epochs = (epochs, epochs)

elif MODE == 'balanced':
    densities = imbal.regression.get_sample_densities(
        y_train,
        kde_bandwidth
    )

    weights = imbal.regression.generate_sample_weights(densities)

history = fit_function(
    x_train,
    y_train,
    sample_weight=weights,
    batch_size=batch_size,
    validation_split=0.2,
    epochs=epochs,
    stratify_batches=STRATIFY,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    ]
)

end = time.time()

print('EXECUTION TIME:', end - start)

if (MODE == 'decoupled'):
    one, two = history
    print('Stage lengths:')
    print(len(one.epoch), len(two.epoch))

print('Evaluating model...')
model.evaluate(x_test, y_test)

predictions = model.predict(x_test)

# kde_bandwidth = imbal.regression.fit_kde(y_train, bin_count=BIN_COUNT)
# imbal.regression.plot_kde_1d(
#     y_train,
#     kde_bandwidth,
#     bin_count=BIN_COUNT,
#     save_figure='sep-c-data-distribution.png' if GEN_OUTPUT else None,
#     padding_factor=0.0001
# )


# plt.scatter(y_test, predictions)
# plt.plot([-10, 10],[-10, 10], linestyle='--', color='red')
# plt.xlabel('Data label')
# plt.ylabel('Prediction')
# plt.xlim(-2, 2)
# plt.ylim(-2, 2)
# plt.savefig(f'fit-comparison-{MODE}-ae-{AE}.png')
# plt.show()
#
# plt.scatter(y_test, predictions)
# plt.plot([-10, 10],[-10, 10], linestyle='--', color='red')
# plt.xlabel('Data label')
# plt.ylabel('Prediction')
# plt.xlim(-2, 2)
# plt.ylim(np.min(predictions)*1.05, np.max(predictions)*1.05)
# plt.show()


if MODEL_TASK == 'classification':
    imbal.classification.tsne_visualization(
        model,
        x_test,
        y_test,
        representation_layer_index=REPRESENTATION_LAYER_INDEX,
        save_figure=f'tsne_visualization-{MODE}-ae-{AE}-rep{REPRESENTATION_LAYER_INDEX}.png' if GEN_OUTPUT else None
    )
else:
    imbal.regression.tsne_visualization(
        model,
        x_test,
        y_test.reshape(-1),
        representation_layer_index=REPRESENTATION_LAYER_INDEX,
        save_figure=f'tsne_visualization-{MODE}-ae-{AE}-rep{REPRESENTATION_LAYER_INDEX}.png' if GEN_OUTPUT else None
    )


predictions = predictions.reshape(-1,)



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_test_labels = y_test.reshape(-1)
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

    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    y_scores = predictions

    fpr, tpr, thresholds = roc_curve(y_test_labels, y_scores, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    print("sklearn AUROC:", roc_auc)

    plt.figure(figsize=(7, 6))
    from matplotlib.collections import LineCollection

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
    y_rare_mask = (y_test_labels <= 18) | (y_test_labels >= 80)
    y_common_mask = (y_test_labels > 18) & (y_test_labels < 80)
    plt.figure(figsize=(7, 6))
    plt.plot([-1, 200], [-1, 200], linestyle="--", linewidth=1, color='black', label="Perfect Prediction")
    plt.scatter(y_test_labels[y_common_mask], predictions.reshape(-1)[y_common_mask], color="blue", alpha=0.3)
    plt.scatter(y_test_labels[y_rare_mask], predictions.reshape(-1)[y_rare_mask], color="green", alpha=0.3)
    plt.plot([-1, 200], [18, 18], color='red', linestyle="--")
    plt.plot([-1, 200], [80, 80], color='red', linestyle="--")
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.xlim(0, 102)
    plt.ylim(0, 102)
    if GEN_OUTPUT:
        plt.savefig(f'regression-true-pred-{MODE}-ae-{AE}-rep{REPRESENTATION_LAYER_INDEX}.png')
    plt.show()

    mask = (y_test > 18) & (y_test < 80)
    rare_mask = (y_test <= 18) | (y_test >= 80)

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
print(history.history['loss'])





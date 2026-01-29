import keras
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os, csv, math, time, imbal

MODEL_TASK = 'regression'

MODE = 'decoupled'
STRATIFY = True
AE = True
REPRESENTATION_LAYER_INDEX = -4
GEN_OUTPUT = True

batch_size = 512
epochs = 860
LEARNING_RATE =2e-4


num_classes = 10

DATASET_PERCENTAGE = 0.8
TRAIN_SPLIT = 0.8

def read_csv_to_list_of_lists(filepath):
    data = []
    with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            data.append(row)
    return data

PATH_START = '/mnt/c/Users/tommy/Desktop/Repos/dr-chan-work-demo'
print(os.getcwd())

def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0

safe_float_vectorized = np.vectorize(safe_float)


# from sklearn.preprocessing import StandardScaler
# data = np.array(read_csv_to_list_of_lists(f'{PATH_START}/tutorials/data/SARCOS/sarcos_inv_training.csv'))
# print(data.shape)
# y_combined = data[1:, -1].astype(float)
# data = safe_float_vectorized(data).astype(float)
# scaler = StandardScaler()
# NUM_FEATURES = data.shape[1] - 1
# x_combined = data[1:, :NUM_FEATURES].astype(float)
# # x_combined = scaler.fit_transform(x_combined)


from sklearn.preprocessing import StandardScaler
data = np.array(read_csv_to_list_of_lists(f'{PATH_START}/tutorials/data/SEP-C/sep_10mev_training.csv'))
print(data.shape)
data = safe_float_vectorized(data[1:]).astype(float)
NUM_FEATURES = 22
y_combined = data[:, NUM_FEATURES].astype(float)
scaler = StandardScaler()
x_combined = data[:, :NUM_FEATURES].astype(float)
x_combined = scaler.fit_transform(x_combined)


# from sklearn.preprocessing import StandardScaler
# data = np.array(read_csv_to_list_of_lists(f'{PATH_START}/tutorials/data/SEP-EC/training/sep_event_1_filled_ie_trim.csv'))[1:]
# for i in range(43):
#     if os.path.exists(f'{PATH_START}/tutorials/data/SEP-EC/training/sep_event_{i+2}_filled_ie_trim.csv'):
#         data = np.concatenate([data, read_csv_to_list_of_lists(f'{PATH_START}/tutorials/data/SEP-EC/training/sep_event_{i+2}_filled_ie_trim.csv')[1:]])
# print(data.shape)
# data = safe_float_vectorized(data).astype(float)
# y_combined = data[:, 182].astype(float)
# scaler = StandardScaler()
# NUM_FEATURES = 182
# x_combined = data[:, :NUM_FEATURES].astype(float)
# x_combined = scaler.fit_transform(x_combined)

print(x_combined.shape)
print(y_combined.shape)

num_data = x_combined.shape[0]
percent_index = int(num_data * DATASET_PERCENTAGE)
shuffled_indices = np.random.RandomState(seed=0).permutation(len(x_combined))[:percent_index]
x_combined = x_combined[shuffled_indices].astype(np.float32)
y_combined = y_combined[shuffled_indices].astype(np.float32)
num_data = x_combined.shape[0]
split_index = int(num_data * TRAIN_SPLIT)
x_train, x_test = x_combined[:split_index], x_combined[split_index:]
y_train, y_test = y_combined[:split_index], y_combined[split_index:]
if MODEL_TASK == 'classification':
    y_train = (y_train >= math.log(10)).astype(int)
    y_test = (y_test >= math.log(10)).astype(int)
print('x_train', x_train.shape)
print('y_train',y_train.shape)
print('x_test',x_test.shape)
print('y_test',y_test.shape)

print(y_train.shape)
print(x_test.shape)

class_split = []
for i in range(num_classes):
    class_split.append(len(y_train[y_train == i]))
print('distribution', class_split)

input_shape = (NUM_FEATURES,)

inputs = keras.Input(shape=input_shape)
x = layers.Dense(18, activation='relu')(inputs)
x = layers.Dense(9, activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(5, activation='relu')(x)
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
    stratify_batches=STRATIFY,
    generate_decoder_branch=AE,
    representation_layer_index=REPRESENTATION_LAYER_INDEX
)
BIN_COUNT=64

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

def determine_ideal_epochs(
    in_model,
    x,
    y,
    sample_weight=None,
    validation_split=0.2,
    delta = 0,
    patience = 10,
    max_epochs=10000
):

    in_model.save_weights('initial.weights.h5')
    TRIALS = 5
    epochs = 0
    for i in range(TRIALS):

        common_x_train = x_train[y_train < math.log(10)]
        rare_x_train = x_train[y_train > math.log(10)]
        common_y_train = y_train[y_train < math.log(10)]
        rare_y_train = y_train[y_train > math.log(10)]

        common_permutation = np.random.permutation(len(common_y_train))
        common_x_train = common_x_train[common_permutation]
        common_y_train = common_y_train[common_permutation]

        rare_permutation = np.random.permutation(len(rare_y_train))
        rare_x_train = rare_x_train[rare_permutation]
        rare_y_train = rare_y_train[rare_permutation]

        common_x_val = common_x_train[round(len(common_x_train)*.8):]
        common_x_train = common_x_train[:round(len(common_x_train) * .8)]
        rare_x_val = rare_x_train[round(len(rare_x_train) * .8):]
        rare_x_train = rare_x_train[:round(len(rare_x_train) * .8)]
        common_y_val = common_y_train[round(len(common_y_train) * .8):]
        common_y_train = common_y_train[:round(len(common_y_train) * .8)]
        rare_y_val = rare_y_train[round(len(rare_y_train) * .8):]
        rare_y_train = rare_y_train[:round(len(rare_y_train) * .8)]

        final_x_train = np.concatenate((common_x_train, rare_x_train))
        final_y_train = np.concatenate((common_y_train, rare_y_train))
        final_x_val = np.concatenate((common_x_val, rare_x_val))
        final_y_val = np.concatenate((common_y_val, rare_y_val))

        train_permutation = np.random.permutation(len(final_x_train))
        val_permutation = np.random.permutation(len(final_x_val))

        final_x_train = final_x_train[train_permutation]
        final_y_train = final_y_train[train_permutation]
        final_x_val = final_x_val[val_permutation]
        final_y_val = final_y_val[val_permutation]

        assert len(final_x_train) == len(final_y_train)
        assert len(final_x_val) == len(final_y_val)

        train_dataset = imbal.classification.DatasetWithBatching(
            final_x_train,
            final_y_train,
            batch_size=batch_size
        )
        val_dataset = imbal.classification.DatasetWithBatching(
            final_x_val,
            final_y_val,
            batch_size=batch_size
        )


        stopper = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=delta,
            patience=patience
        )
        in_model.compile(
            loss="binary_crossentropy" if MODEL_TASK == 'classification' else 'mse',
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            metrics=["accuracy" if MODEL_TASK == 'classification' else "mse"],
            stratify_batches=STRATIFY,
            generate_decoder_branch=AE,
            representation_layer_index=REPRESENTATION_LAYER_INDEX
        )
        history = in_model.fit(
            train_dataset,
            batch_size=batch_size,
            validation_data=val_dataset,
            sample_weight=sample_weight,
            epochs=max_epochs,
            callbacks=[stopper]
        )
        epochs += len(history.history['loss']) - patience
        in_model.load_weights('initial.weights.h5')

    return round(epochs / TRIALS)


# print(f'\n\n\nTESTING\n\n{determine_ideal_epochs(
#     model,
#     x_train,
#     y_train,
#     patience=20,
#     delta=0
# )}\n\n\n')

history = None

start = time.time()
weights = np.ones(x_train.shape[0])
if MODE == 'decoupled':
    bandwidth = imbal.regression.fit_kde(
        y_train,
        bin_count=BIN_COUNT
    )
    densities = imbal.regression.get_sample_densities(
        y_train,
        bandwidth
    )
    weights = imbal.regression.generate_sample_weights(densities)

elif MODE == 'balanced':
    bandwidth = imbal.regression.fit_kde(
        y_train,
        bin_count=BIN_COUNT
    )
    densities = imbal.regression.get_sample_densities(
        y_train,
        bandwidth
    )
    weights = imbal.regression.generate_sample_weights(densities)

history = fit_function(
    x_train,
    y_train,
    sample_weight=weights,
    batch_size=batch_size,
    epochs=epochs
)

end = time.time()

print('EXECUTION TIME:', end - start)

print('Evaluating model...')
model.evaluate(x_test, y_test)

predictions = model.predict(x_test)

kde_bandwidth = imbal.regression.fit_kde(y_combined, bin_count=BIN_COUNT)
imbal.regression.plot_kde_1d(
    y_combined,
    kde_bandwidth,
    bin_count=BIN_COUNT,
    save_figure='sep-ec-kde-curve.png' if GEN_OUTPUT else None
)


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



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    y_scores = predictions

    fpr, tpr, thresholds = roc_curve(y_test_labels, y_scores, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    print("sklearn AUROC:", roc_auc)

    plt.figure(figsize=(7, 6))
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import numpy as np

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
    plt.scatter(y_test_labels[y_common_mask], predictions.reshape(-1)[y_common_mask], color="blue", alpha=0.3)
    plt.scatter(y_test_labels[y_rare_mask], predictions.reshape(-1)[y_rare_mask], color="green", alpha=0.3)
    plt.plot([-10, 10], [math.log(10), math.log(10)], color='red', linestyle="--")
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
print(history.history['loss'])





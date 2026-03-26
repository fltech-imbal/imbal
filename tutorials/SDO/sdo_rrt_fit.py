"""
Import packages
"""
import imbal
import os, glob, math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras import layers, optimizers

"""
Load data
"""
SDO_DATA_PATH = '../data/SDOBenchmark'
TRAINING_DATA_MAX_SIZE = 50
TESTING_DATA_MAX_SIZE = 50

def load_sdo_data(data_path, max_samples=None):
    df = pd.read_csv(os.path.join(data_path, 'meta_data.csv'))
    good_file_paths = []
    good_file_fluxes = []
    for i in range(len(df)):
        timestamp_id = df['id'][i]
        log_peak_flux = math.log10(float(df['peak_flux'][i]))
        print(f'Finding data from "{data_path}" [{i+1}/{len(df["id"])}]', end='\r')
        timestamp_portions = str(timestamp_id).split('_')
        folder_path = str(os.path.join(data_path, timestamp_portions[0]))
        sub_folder_path = str(os.path.join(folder_path, '_'.join(timestamp_portions[1:])))
        if not os.path.exists(sub_folder_path):
            continue

        folder_timestamp = datetime.strptime("_".join(timestamp_portions[-4:-1]), '%H_%M_%S')
        minus_ten_minutes = folder_timestamp - timedelta(minutes=10)
        images = glob.glob(os.path.join(sub_folder_path, '*.jpg'))

        def within_five_seconds(image_name, timestamp):
            image_time_string = image_name.split('T')[1][:6]
            image_timestamp = datetime.strptime(image_time_string, '%H%M%S')
            return abs((timestamp - image_timestamp).total_seconds()) < 5

        minus_ten_images = [x for x in images if within_five_seconds(x, minus_ten_minutes)]
        if len(minus_ten_images) == 10:
            good_file_paths.append(minus_ten_images)
            good_file_fluxes.append(log_peak_flux)

        if max_samples is not None and len(good_file_paths) == max_samples:
            print('\nFound maximum number of samples. Stopping early.',end='\r')
            break


    loaded_images = np.zeros((len(good_file_paths), 256, 256, 10))
    loaded_data_fluxes = np.array(good_file_fluxes)

    print()
    for index, image_paths in enumerate(good_file_paths):
        print(f'Loading SDO samples [{index+1}/{len(good_file_paths)}]', end='\r')
        image_list = [Image.open(x).convert('L') for x in image_paths]
        stacked_images = np.stack(image_list, axis=-1)
        loaded_images[index] = stacked_images / 255.0

    print(f'\n{len(good_file_paths)} data samples loaded successfully')
    return loaded_images, loaded_data_fluxes

x_train, y_train = load_sdo_data(os.path.join(SDO_DATA_PATH, 'training'), max_samples=TRAINING_DATA_MAX_SIZE)
x_test, y_test = load_sdo_data(os.path.join(SDO_DATA_PATH, 'test'), max_samples=TESTING_DATA_MAX_SIZE)

print(
    f'Loaded data with the following shapes:\n'
    f'\tx_train: {x_train.shape}\n'
    f'\ty_train: {y_train.shape}\n'
    f'\tx_test: {x_test.shape}\n'
    f'\ty_test {y_test.shape}'
)

"""
Calculate data density distribution, and extract sample densities
"""
KDE_BIN_COUNT=32

data_kde_bandwidth = imbal.regression.fit_kde(y_train, bin_count=KDE_BIN_COUNT)
sample_densities = imbal.regression.get_sample_densities(y_train, data_kde_bandwidth)


"""
Build model
"""
def build_simple_cnn():
    input_layer = layers.Input((256, 256, 10))
    x = layers.Conv2D(32, 3, activation='relu')(input_layer)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(1)(x)

    model = imbal.regression.Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    return model

model = build_simple_cnn()

"""
Compile and train model
"""
LEARNING_RATE = 5e-5
EPOCHS = 50
BATCH_SIZE = 32

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    metrics=['mae'],
    representation_layer_index=-2
)

model.rRT_fit(
    x_train,
    y_train,
    sample_density=sample_densities,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    stratify_batches=True
)

model.evaluate(x_test, y_test)

"""
Data and results visualization
"""

train_rare_mask = y_train > -4
test_rare_mask = y_test > -4
print('Number of rare training samples:', np.sum(train_rare_mask.astype(np.int32)))
print('Number of rare testing samples:', np.sum(test_rare_mask.astype(np.int32)))

train_predictions = model.predict(x_train).reshape(-1)
test_predictions = model.predict(x_test).reshape(-1)

train_predictions_rare = train_predictions[train_rare_mask]
train_labels_rare = y_train[train_rare_mask]
test_predictions_rare = test_predictions[test_rare_mask]
test_labels_rare = y_test[test_rare_mask]

overall_train_mae = np.mean(np.abs(train_predictions - y_train))
rare_train_mae = np.mean(np.abs(train_predictions_rare - train_labels_rare))

overall_test_mae = np.mean(np.abs(test_predictions - y_test))
rare_test_mae = np.mean(np.abs(test_predictions_rare - test_labels_rare))

print(
    f'Overall train MAE: {overall_train_mae:.3f}\n'
    f'Rare train MAE: {rare_train_mae:.3f}\n'
    f'Overall test MAE: {overall_test_mae:.3f}\n'
    f'Rare test MAE: {rare_test_mae:.3f}'
)


imbal.regression.plot_kde_1d(
    y_train,
    data_kde_bandwidth,
    bin_count=KDE_BIN_COUNT,
    show_bin_count=False,
    save_figure='sample-sdo-rrt-fit-data-distribution.png'
)

def plot_true_vs_predictions(
    labels,
    predictions,
    rare_threshold=-4,
    low_bound=-9.5,
    high_bound=-2,
    save_figure=None
):
    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)

    rare_mask = labels > rare_threshold
    frequent_mask = ~rare_mask

    frequent_labels = labels[frequent_mask]
    frequent_predictions = predictions[frequent_mask]
    rare_labels = labels[rare_mask]
    rare_predictions = predictions[rare_mask]

    plt.figure(figsize=(7, 6))
    plt.plot([low_bound, high_bound], [low_bound, high_bound], linestyle="--", linewidth=1, color='black', label="Perfect Prediction")
    light_gray = '#BBBBBB'
    plt.plot([rare_threshold, rare_threshold], [low_bound, high_bound], linestyle="--", linewidth=1, color=light_gray)
    plt.plot([low_bound, high_bound], [rare_threshold, rare_threshold], linestyle="--", linewidth=1, color=light_gray)
    plt.scatter(frequent_labels, frequent_predictions, color="#00FF00", alpha=0.3)
    plt.scatter(rare_labels, rare_predictions, color="#FF0000", alpha=0.2)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.xlim(low_bound, high_bound)
    plt.ylim(low_bound, high_bound)
    if save_figure is not None:
        plt.savefig(save_figure)
    plt.show()

plot_true_vs_predictions(
    y_test,
    test_predictions,
    save_figure='sample-sdo-rrt-fit-label-vs-prediction-plot.png'
)
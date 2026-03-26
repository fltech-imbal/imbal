import os, glob, math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from PIL import Image
import imbal

SDO_DATA_PATH = '../data/SDOBenchmark'
TRAINING_DATA_MAX_SIZE = None
TESTING_DATA_MAX_SIZE = None

def load_sdo_data(data_path, max_samples=None):
    df = pd.read_csv(os.path.join(data_path, 'meta_data.csv'))
    good_file_paths = []
    good_file_fluxes = []
    for i in range(len(df)):
        timestamp_id = df['id'][i]
        log_peak_flux = math.log10(float(df['peak_flux'][i]))
        print(f'Finding data from "{data_path}" [{i+1}/{len(df["id"])}]', end='\n')
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
            print('\nFound maximum number of samples. Stopping early.',end='\n')
            break


    loaded_images = np.zeros((len(good_file_paths), 256, 256, 10), dtype=np.uint8)
    loaded_data_fluxes = np.array(good_file_fluxes)

    print()
    for index, image_paths in enumerate(good_file_paths):
        print(f'Loading SDO samples [{index+1}/{len(good_file_paths)}]', end='\n')
        image_list = [Image.open(x).convert('L') for x in image_paths]
        stacked_images = np.stack(image_list, axis=-1)
        loaded_images[index] = stacked_images

    print(f'\n{len(good_file_paths)} data samples loaded successfully')
    return loaded_images, loaded_data_fluxes

x_train, y_train = load_sdo_data(os.path.join(SDO_DATA_PATH, 'training'), max_samples=TRAINING_DATA_MAX_SIZE)

(_, _), (x_sub, y_sub) = imbal.regression.split(x_train, y_train, test_size=0.20)

def save(images, labels, path, max_num=1000):
    with open(f"{path}/log_peak_flux.txt", "w") as file:
        for index, label in enumerate(labels.reshape(-1)):
            if index == max_num:
                break
            file.write(f"{label}\n")
            for i in range(10):
                img = Image.fromarray(images[index, :, :, i].reshape(256, 256))
                img.save(f'{path}/sdo_subset_sample_{index}_image_{i}.jpg')

save(x_sub, y_sub, 'sub-sdo/training')

x_test, y_test = load_sdo_data(os.path.join(SDO_DATA_PATH, 'test'), max_samples=TESTING_DATA_MAX_SIZE)

(_, _), (x_test_sub, y_test_sub) = imbal.regression.split(x_test, y_test, test_size=0.50)

save(x_test_sub, y_test_sub, 'sub-sdo/test', max_num=300)
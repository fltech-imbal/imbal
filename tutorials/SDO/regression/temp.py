import os
import numpy as np
from PIL import Image

import PIL

"""
Load data
"""
SDO_DATA_PATH = '../../data/SDOBenchmark' # Ensure data is located at this path

def load_sdo_data(data_path):
    # Load labels (log peak flux)
    with open(os.path.join(data_path, 'log_peak_flux.txt'), 'r') as file:
        contents = file.read().strip()
        loaded_data_fluxes = np.array([float(x) for x in contents.split('\n')])

    # Load images (10 images per sample, 256x256 per image)
    loaded_images = np.zeros((len(loaded_data_fluxes), 128, 128, 10), dtype=np.float32)
    for i in range(len(loaded_data_fluxes)):
        print(f'Loading SDO samples [{i+1}/{len(loaded_data_fluxes)}]', end='\r')
        image_list = [Image.open(os.path.join(data_path, f'sdo_subset_sample_{i}_image_{x}.jpg')).convert('L').resize((128,128), Image.Resampling.BILINEAR) for x in range(10)]
        stacked_images = np.stack(image_list, axis=-1) # Images stacked along channels
        loaded_images[i] = stacked_images / 255.0 # Normalize black and white pixel values from 0 to 1

    print(f'\n{len(loaded_data_fluxes)} data samples loaded successfully')

    print('Saving resized images...')
    for i in range(len(loaded_data_fluxes)):
        print(f'Saving SDO samples [{i + 1}/{len(loaded_data_fluxes)}]', end='\r')
        for x in range(10):
            pixel_data = (loaded_images[i, :, :, x] * 255.0).astype(np.uint8)
            img = Image.fromarray(pixel_data, mode='L')
            temp = data_path.split('/')[-1]
            img.save(os.path.join(f'temp/{temp}', f'sdo_subset_sample_{i}_image_{x}.jpg'))
    print(f'\n{len(loaded_data_fluxes)} samples saved successfully')

    return loaded_images, loaded_data_fluxes

# Load train and test data via function defined above
x_train, y_train = load_sdo_data(os.path.join(SDO_DATA_PATH, 'training'))
x_test, y_test = load_sdo_data(os.path.join(SDO_DATA_PATH, 'test'))

print(
    f'Loaded data with the following shapes:\n'
    f'\tx_train: {x_train.shape}\n'
    f'\ty_train: {y_train.shape}\n'
    f'\tx_test: {x_test.shape}\n'
    f'\ty_test {y_test.shape}'
)
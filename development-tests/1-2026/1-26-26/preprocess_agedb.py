import os
import glob
import numpy as np
import cv2
import pickle

from PIL import Image

PATH_START = '/mnt/c/Users/tommy/Desktop/Repos/dr-chan-work-demo'
PADDING = 10
ADDITIONAL_TOP_PADDING = 50
SCALE_FACTOR = 2

age_db_path = os.path.join(PATH_START, 'AgeDB/03_Protocol_Images/03_Protocol_Images')
save_output_path = os.path.join(PATH_START, 'AgeDB/cropped')

# age_labels = []
#
# images = []
# image_sizes = []
#
# age_db_images = glob.glob(os.path.join(age_db_path, '*.jpg'))
# for index, image_path in enumerate(age_db_images):
#     print(f"Processing images [{index+1}/{len(age_db_images)}]...      ", end='\r')
#     points_path = image_path.replace('.jpg', '.pts')
#     if not os.path.exists(points_path):
#         continue
#
#     image_array = np.array(Image.open(image_path))
#
#     with open(points_path, 'r') as f:
#         lines = [line.strip() for line in f.readlines()][3:-1]
#         points = np.array([[float(value) for value in line.split(' ')] for line in lines])
#         min_x = int(max(points[:, 0].min() - PADDING, 0))
#         max_x = int(min(points[:, 0].max() + PADDING, image_array.shape[1]))
#         min_y = int(max(points[:, 1].min() - PADDING - ADDITIONAL_TOP_PADDING, 0))
#         max_y = int(min(points[:, 1].max() + PADDING, image_array.shape[0]))
#
#     cropped_image = image_array[min_y:max_y, min_x:max_x]
#
#     age = int(image_path.split('/')[-1][:-4].split('_')[-1])
#     age_labels.append(age)
#     images.append(cropped_image)
# print()
#
# age_labels = np.array(age_labels)
# np.save(os.path.join(save_output_path, 'age_labels.npy'), age_labels)

print('loading pickle...')
cropped_pickle_path = os.path.join(save_output_path, 'cropped_images.pkl')
with open(cropped_pickle_path, 'rb') as f:
    images = pickle.load(f)

print('finding image sizes...')
image_sizes = [image.shape[:2] for image in images]

average_dims = np.mean(np.array(image_sizes), axis=0)
average_y = round(average_dims[0])
average_x = round(average_dims[1])

print('resizing images...')

images_resized = np.array([
    cv2.resize(img, (average_x // SCALE_FACTOR, average_y // SCALE_FACTOR)) if img.ndim == 3
    else cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (average_x // SCALE_FACTOR, average_y // SCALE_FACTOR))
    for img in images
])

print('saving resized image pickle...')
np.save(os.path.join(save_output_path, 'cropped_resized_images.npy'), images_resized)
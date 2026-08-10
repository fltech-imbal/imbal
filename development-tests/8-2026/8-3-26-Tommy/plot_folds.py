import pandas as pd
import glob
import numpy as np
import os
from matplotlib import pyplot as plt

TIME_SERIES_DATA_PATH = 'dtw-data'
CLUSTER_SIZE = 5

# def load_time_series(path):
#     series = []
#     files = glob.glob(path + '/*.csv')
#     for file in files:
#         df = pd.read_csv(file)
#         series.append(df)
#     return series

fold = 1
path = f'{TIME_SERIES_DATA_PATH}/fold-{fold}'

while os.path.exists(path):
    fold_time_series = glob.glob(path + '/*.csv')
    for file in fold_time_series:
        df = pd.read_csv(file)
        intensities = df['Proton Intensity'].to_numpy()
        x = np.arange(len(intensities))
        plt.plot(x, intensities)
        plt.title(f'Fold {fold} - {file}')
        plt.show()

    fold += 1
    path = f'{TIME_SERIES_DATA_PATH}/fold-{fold}'

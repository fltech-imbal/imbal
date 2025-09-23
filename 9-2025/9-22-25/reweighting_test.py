import imbal.regression.generate_weights
from imbal.classification import generate_weights as generate_classification_weights
from imbal.regression import generate_weights as generate_regression_weights
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import csv

def read_csv_to_list_of_lists(filepath):
    data = []
    with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            data.append(row)
    return data

# Download latest version
path = kagglehub.dataset_download("arashnic/imbalanced-data-practice")
data = np.array(read_csv_to_list_of_lists(path + '/aug_train.csv'))
data = data[1:1000, 2].astype(np.float64)
# print(data)

labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2])

print(generate_classification_weights(labels, {
    0: 1/6,
    1: 1/3,
    2: 1/2
}).tolist())

print(generate_classification_weights(labels).tolist())

labels = np.array([0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.12, 0.13, 0.15, 0.9, 0.95, 1, 1.02, 1.04, 1.05, 1.1, 1.12, 2.1, 2.02])
weights, kde = generate_regression_weights(data, return_kde=True, mode=imbal.regression.RegressionWeightMode.AVERAGE, bin_count=20)
print(weights.tolist())

# Create a range of values over which to evaluate the KDE
x_plot = np.linspace(data.min() - 1, data.max() + 1, 1000).reshape(-1, 1)
log_dens = kde.score_samples(x_plot)
plt.figure(figsize=(8, 6))
plt.plot(x_plot, np.exp(log_dens), label='KDE Curve')
plt.hist(data, bins=20, density=True, alpha=0.6, label='Histogram')
plt.title('Kernel Density Estimation')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
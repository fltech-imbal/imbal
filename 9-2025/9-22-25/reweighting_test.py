from imbal.classification import generate_weights as generate_classification_weights
from imbal.regression import generate_weights as generate_regression_weights
import numpy as np
import matplotlib.pyplot as plt

labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2])

print(generate_classification_weights(labels, {
    0: 1/6,
    1: 1/3,
    2: 1/2
}).tolist())

print(generate_classification_weights(labels).tolist())

second_test_labels = labels
weights, kde = generate_regression_weights(second_test_labels, return_kde=True)
print(weights.tolist())

# Create a range of values over which to evaluate the KDE
x_plot = np.linspace(second_test_labels.min() - 1, second_test_labels.max() + 1, 1000).reshape(-1, 1)
log_dens = kde.score_samples(x_plot)
plt.figure(figsize=(8, 6))
plt.plot(x_plot, np.exp(log_dens), label='KDE Curve')
plt.hist(second_test_labels, bins=30, density=True, alpha=0.6, label='Histogram')
plt.title('Kernel Density Estimation')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
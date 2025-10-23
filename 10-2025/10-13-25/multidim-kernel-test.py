import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import imbal

data = np.random.multivariate_normal([0, 0], [[-1, .9],[.9,-1]], size=5000)

print(data[:20])

x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
xx, yy = np.meshgrid(x, y)
grid = np.vstack([xx.ravel(), yy.ravel()]).T
print(grid.shape)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
ax.hist2d(data[:, 0], data[:, 1], cmap='viridis', bins=50)

# Make it look nice
ax.set_xlabel('X')
ax.set_ylabel('Y')
# ax.set_zlabel('Density')
ax.set_title('3D Kernel Density Estimate Surface')

plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
BINS=32

print(1)
bandwidth = imbal.regression.kde.fit_kde(
    data,
    bandwidth='kl_divergence',
    bin_count=BINS,
    steps_per_bin=2
)
print(2)

kde = KernelDensity(bandwidth=bandwidth).fit(data)

print(grid.shape)
values = np.exp(kde.score_samples(grid))

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
ax.scatter(grid[:, 0], grid[:, 1], c=values, cmap='viridis')

# Make it look nice
ax.set_xlabel('X')
ax.set_ylabel('Y')
# ax.set_zlabel('Density')
ax.set_title('3D Kernel Density Estimate Surface')
plt.show()
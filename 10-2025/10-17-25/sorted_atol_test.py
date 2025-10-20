import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import imbal
import time

# data = np.random.multivariate_normal([0, 0], [[-.2, .5],[.5,-1]], size=30000)
#
# x = np.linspace(-3, 3, 200)
# y = np.linspace(-3, 3, 200)
# xx, yy = np.meshgrid(x, y)
# grid = np.vstack([xx.ravel(), yy.ravel()]).T
#
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111)
# counts, xedges, yedges, image = ax.hist2d(data[:, 0], data[:, 1], cmap='viridis', bins=50, density=True)
# bin_width_x = np.diff(xedges)[0]
# bin_width_y = np.diff(yedges)[0]
# bin_area = bin_width_x * bin_width_y
# density = counts / (counts.sum() * bin_area)
#
# max_density = density.max()
# image.set_clim(vmin=0, vmax=max_density)
# plt.colorbar(image, ax=ax)
#
#
# # Make it look nice
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)
# # ax.set_zlabel('Density')
# ax.set_title('3D Kernel Density Estimate Surface')
# # ax.view_init(elev=35, azim=135)  # tweak for a nice angle
# plt.savefig('true-2d-distribution.png')
# plt.show()
#
# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
data = np.random.normal(size=30000)

BINS=32

print(1)
bandwidth = imbal.regression.kde.fit_kde(
    data,
    bandwidth='kl_divergence',
    bin_count=BINS,
    steps_per_bin=10
)
print(2)

total = 0
for i in range(10):
    start = time.time()
    densities = imbal.regression.get_densities(
        data,
        bandwidth=bandwidth,
        atol=1e-4
    )
    end = time.time()
    total += end - start
print('regular', total/10)

data = np.sort(data)

total = 0
for i in range(10):
    start = time.time()
    densities = imbal.regression.get_densities(
        data,
        bandwidth=bandwidth,
        atol=1e-4
    )
    end = time.time()
    total += end - start
print('regular', total/10)
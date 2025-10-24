import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import imbal
import time

data = np.append(np.random.multivariate_normal([0, 0], [[1, 0], [0, .15]], size=15000), [[-2.99, -2.99],[2.99, 2.99]], axis=0)

print(data.shape)

spacing=100
x = np.linspace(-3, 3, spacing)
y = np.linspace(-3, 3, spacing)
xx, yy = np.meshgrid(x, y)
grid = np.vstack([xx.ravel(), yy.ravel()]).T

x = np.linspace(-3, 3, spacing//2)
y = np.linspace(-3, 3, spacing//2)
xx, yy = np.meshgrid(x, y)
half_grid = np.vstack([xx.ravel(), yy.ravel()]).T

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
counts, xedges, yedges, image = ax.hist2d(data[:, 0], data[:, 1], cmap='viridis', bins=50, density=True)
bin_width_x = np.diff(xedges)[0]
bin_width_y = np.diff(yedges)[0]
bin_area = bin_width_x * bin_width_y
density = counts / (counts.sum() * bin_area)

max_density = density.max()
image.set_clim(vmin=0, vmax=max_density)
plt.colorbar(image, ax=ax)


# Make it look nice
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
# ax.set_zlabel('Density')
ax.set_title('3D Kernel Density Estimate Surface')
# ax.view_init(elev=35, azim=135)  # tweak for a nice angle
plt.savefig('true-2d-distribution.png')
plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
BINS=32
COMPARE_DATA = data

print(1)
bandwidth = imbal.regression.kde.fit_kde(
    data,
    bin_count=BINS,
    steps_per_bin=2
)
print(2)

kde = KernelDensity(bandwidth=bandwidth).fit(data)

start = time.time()
densities = imbal.regression.get_densities(
    COMPARE_DATA,
     bandwidth=bandwidth
)
# densities = np.exp(kde.score_samples(data)).reshape(-1, 1)
# less_densities = np.exp(kde.score_samples(half_grid))

end = time.time()
print('regular', end-start)
# print(densities.shape)
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111)
# sc = ax.scatter(half_grid[:, 0], half_grid[:, 1], c=less_densities, cmap='viridis')
# sc.set_clim(vmin=0, vmax=max_density)
# plt.colorbar(sc, ax=ax)
# # Make it look nice
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)
# # ax.set_zlabel('Density')
# ax.set_title('3D Kernel Density Estimate Surface')
# # ax.view_init(elev=35, azim=135)  # tweak for a nice angle
# plt.savefig('per-bin-2d-kde-scatter.png')
# plt.show()
#
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111)
# sc = ax.scatter(COMPARE_DATA[:, 0], COMPARE_DATA[:, 1], c=densities, cmap='viridis')
# sc.set_clim(vmin=0, vmax=max_density)
# plt.colorbar(sc, ax=ax)
# # Make it look nice
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)
# # ax.set_zlabel('Density')
# ax.set_title('3D Kernel Density Estimate Surface')
# # ax.view_init(elev=35, azim=135)  # tweak for a nice angle
# plt.savefig('2d-kde-scatter.png')
# plt.show()


print(3)
start = time.time()
lin_int_densities, lin_int_approx = imbal.regression.get_densities(
    COMPARE_DATA,
    bandwidth=bandwidth,
    interpolation_samples=2*BINS,
    interpolation_method='linear',
    return_interpolation_samples=True
)
end=time.time()
print('lin_int', end-start)
print(lin_int_densities.shape)

print(4)
start=time.time()
loc_approx_densities, loc_approx_approx = imbal.regression.get_densities(
    COMPARE_DATA,
    bandwidth,
    atol=1e-4,
    return_interpolation_samples=True
)
end=time.time()
print('loc_approx', end-start)
print(loc_approx_densities.shape)

num_bins = 32  # Number of desired logarithmic bins
lin_errors = np.abs(lin_int_densities - densities).reshape(-1)
bins = np.logspace(np.log10(min(lin_errors)), np.log10(max(lin_errors)), num_bins + 1)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), constrained_layout=True)
ax[0].hist(lin_errors, bins=bins, alpha=0.6)
ax[0].set_xscale('log')
ax[0].set_title('Errors of linear_interpolation')
ax[0].set_ylim(auto=True)
loc_approx_errors = np.abs(loc_approx_densities - densities).reshape(-1)
# bins = np.logspace(np.log10(min(loc_approx_errors)), np.log10(max(loc_approx_errors)), num_bins + 1)
bins = np.linspace(0, max(loc_approx_errors), num_bins + 1)
ax[1].hist(loc_approx_errors, bins=bins, alpha=0.6)
# ax[1].set_xscale('log')
ax[1].set_title('Errors of local_approximation')
ax[1].set_ylim(auto=True)
plt.savefig('2d-data-error-histogram.png')
plt.show()

densities = densities.reshape(-1)

print(densities.shape)
print(lin_errors.shape)
print(loc_approx_errors.shape)
print('lin_int average error:', np.mean(lin_errors))
print('loc_approx average error:', np.mean(loc_approx_errors))
print(lin_errors.shape)
print(loc_approx_errors.shape)
print('lin_int average % error:', np.mean(lin_errors / densities * 100))
print('loc_approx average % error:', np.mean(loc_approx_errors / densities * 100))
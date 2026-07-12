import numpy as np
import matplotlib.pyplot as plt

def plot_representation_space(
    representation_vectors,
    labels,
    dim_one_index=0,
    dim_two_index=1,
    save_figure=None,
    vmin=-2,
    vmax=2,
    margin=0.1
):
    dim_one_sort_order = np.argsort(representation_vectors[:, dim_one_index].reshape(-1))
    dim_two_sort_order = np.argsort(representation_vectors[:, dim_two_index].reshape(-1))

    min_rep_x = np.min(representation_vectors[:, dim_one_index][dim_one_sort_order[5]])
    max_rep_x = np.max(representation_vectors[:, dim_one_index][dim_one_sort_order[-5]])
    min_rep_y = np.min(representation_vectors[:, dim_two_index][dim_two_sort_order[5]])
    max_rep_y = np.max(representation_vectors[:, dim_two_index][dim_two_sort_order[-5]])

    abs_labels = np.abs(labels - float(np.median(labels)))
    sort_indices = np.argsort(abs_labels)
    plot_sorted_labels = labels[sort_indices]
    plot_sorted_representations = representation_vectors[sort_indices]

    distance_rep_x = max_rep_x - min_rep_x
    distance_rep_y = max_rep_y - min_rep_y
    if distance_rep_x > distance_rep_y:
        diff = distance_rep_x - distance_rep_y
        half_diff = diff / 2
        min_rep_y -= half_diff
        max_rep_y += half_diff
    else:
        diff = distance_rep_y - distance_rep_x
        half_diff = diff / 2
        min_rep_x -= half_diff
        max_rep_x += half_diff

    min_rep_x -= margin
    max_rep_x += margin
    min_rep_y -= margin
    max_rep_y += margin

    plt.scatter(
        plot_sorted_representations[:, dim_one_index].reshape(-1),
        plot_sorted_representations[:, dim_two_index].reshape(-1),
        c=plot_sorted_labels,
        cmap='jet',
        vmin=vmin,
        vmax=vmax,
        alpha=0.5
    )
    plt.xlim(min_rep_x, max_rep_x)
    plt.ylim(min_rep_y, max_rep_y)
    plt.colorbar()
    if save_figure is not None:
        plt.savefig(save_figure)
    plt.show()

def plot_representation_space_3d(
    representation_vectors,
    labels,
    dim_one_index=0,
    dim_two_index=1,
    dim_three_index=2,
    save_figure=None,
    vmin=-2,
    vmax=2,
    margin=0.25,
    exclude_extremes=2
):
    # Sort each dimension to ignore a few extreme outliers
    dim_one_sort_order = np.argsort(representation_vectors[:, dim_one_index])
    dim_two_sort_order = np.argsort(representation_vectors[:, dim_two_index])
    dim_three_sort_order = np.argsort(representation_vectors[:, dim_three_index])

    min_rep_x = representation_vectors[:, dim_one_index][dim_one_sort_order[exclude_extremes]]
    max_rep_x = representation_vectors[:, dim_one_index][dim_one_sort_order[-exclude_extremes]]

    min_rep_y = representation_vectors[:, dim_two_index][dim_two_sort_order[exclude_extremes]]
    max_rep_y = representation_vectors[:, dim_two_index][dim_two_sort_order[-exclude_extremes]]

    min_rep_z = representation_vectors[:, dim_three_index][dim_three_sort_order[exclude_extremes]]
    max_rep_z = representation_vectors[:, dim_three_index][dim_three_sort_order[-exclude_extremes]]

    # Make axes roughly equal length
    x_range = max_rep_x - min_rep_x
    y_range = max_rep_y - min_rep_y
    z_range = max_rep_z - min_rep_z

    max_range = max(x_range, y_range, z_range)

    def expand_axis(min_val, max_val):
        center = (min_val + max_val) / 2
        half = max_range / 2
        return center - half - margin, center + half + margin

    min_rep_x, max_rep_x = expand_axis(min_rep_x, max_rep_x)
    min_rep_y, max_rep_y = expand_axis(min_rep_y, max_rep_y)
    min_rep_z, max_rep_z = expand_axis(min_rep_z, max_rep_z)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    common_mask = (labels > -0.5) & (labels < 0.5)

    scatter = ax.scatter(
        representation_vectors[common_mask][:, dim_one_index],
        representation_vectors[common_mask][:, dim_two_index],
        representation_vectors[common_mask][:, dim_three_index],
        c=labels[common_mask],
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        alpha=0.5,
    )

    scatter = ax.scatter(
        representation_vectors[~common_mask][:, dim_one_index],
        representation_vectors[~common_mask][:, dim_two_index],
        representation_vectors[~common_mask][:, dim_three_index],
        c=labels[~common_mask],
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        alpha=0.5,
    )

    ax.set_xlim(min_rep_x, max_rep_x)
    ax.set_ylim(min_rep_y, max_rep_y)
    ax.set_zlim(min_rep_z, max_rep_z)

    ax.set_xlabel(f"Dimension {dim_one_index}")
    ax.set_ylabel(f"Dimension {dim_two_index}")
    ax.set_zlabel(f"Dimension {dim_three_index}")

    fig.colorbar(scatter, ax=ax)

    if save_figure is not None:
        plt.savefig(save_figure, dpi=300, bbox_inches="tight")

    plt.show()
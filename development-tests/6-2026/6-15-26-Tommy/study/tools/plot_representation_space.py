import numpy as np
import matplotlib.pyplot as plt

def plot_representation_space(representation_vectors, labels, dim_one_index, dim_two_index):
    min_rep_x = np.min(representation_vectors[:, dim_one_index])
    max_rep_x = np.max(representation_vectors[:, dim_one_index])
    min_rep_y = np.min(representation_vectors[:, dim_two_index])
    max_rep_y = np.max(representation_vectors[:, dim_two_index])

    abs_labels = np.abs(labels)
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

    min_rep_x -= 1
    max_rep_x += 1
    min_rep_y -= 1
    max_rep_y += 1

    plt.scatter(
        plot_sorted_representations[:, dim_one_index].reshape(-1),
        plot_sorted_representations[:, dim_two_index].reshape(-1),
        c=plot_sorted_labels,
        cmap='jet',
        alpha=0.5
    )
    plt.xlim(min_rep_x, max_rep_x)
    plt.ylim(min_rep_y, max_rep_y)
    plt.colorbar()
    plt.show()
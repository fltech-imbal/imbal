import imbal
import numpy as np

def generate_weights(
    x_train,
    y_train,
    x_val,
    y_val,
    weight_alpha=0.6,
    combine_validation=True
):
    if not combine_validation:
        kde_bandwidth = imbal.regression.fit_kde(
            y_train,
            bin_count=64
        )

        sample_densities = imbal.regression.get_sample_densities(
            y_train,
            kde_bandwidth,
        )
        sample_weights = imbal.regression.reciprocal_importance(sample_densities, alpha=weight_alpha)
        val_densities = imbal.regression.get_sample_densities(
            y_val,
            kde_bandwidth,
            distribution=y_train
        )
        w_val = imbal.regression.reciprocal_importance(val_densities, alpha=weight_alpha)
        val_data = (x_val, y_val, w_val)
    else:
        x_train = np.concatenate((x_train, x_val))
        y_train = np.concatenate((y_train, y_val))

        kde_bandwidth = imbal.regression.fit_kde(
            y_train,
            bin_count=64
        )

        sample_densities = imbal.regression.get_sample_densities(
            y_train,
            kde_bandwidth,
        )
        sample_weights = imbal.regression.reciprocal_importance(sample_densities)

        val_data = None

    return (x_train, y_train, sample_weights), val_data
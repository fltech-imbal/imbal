import imbal.regression as imbal
def labels_to_kde_weights(
        labels,
        optimization=None,
        steps_per_bin=10,
        bin_count=64,
        average_samples_per_bin=None,
        padding_factor=0.01,
        plot_kde=False,
        save_figure=None,
        use_axes=None,
        bandwidth='mse',
        fine_search = 10,
        tolerance = 1e-3,
        return_kde=False
):
    kde = imbal.kde.fit_kde(
        labels,
        bandwidth=bandwidth,
        bin_count=bin_count,
        steps_per_bin=steps_per_bin,
        average_samples_per_bin=average_samples_per_bin,
        padding_factor=padding_factor,
        fine_search=fine_search,
        tolerance=tolerance,
    )

    weights, approx = imbal.generate_weights(
        labels,
        density_mapping=kde,
        bin_count=bin_count,
        optimization=optimization,
        return_optimization=True,
        steps_per_bin=steps_per_bin,
        padding_factor=padding_factor
    )


    if plot_kde or use_axes or save_figure:
        imbal.kde.plot_kde(
            labels,
            kde,
            bin_count=bin_count,
            approximation=approx,
            padding_factor = padding_factor,
            use_axes=use_axes,
            save_figure=save_figure
        )

    if return_kde:
        return weights, kde
    else:
        return weights
import imbal.regression as imbal
def labels_to_kde_weights(
    labels,
    interpolation_method=None,
    interpolation_samples=None,
    atol=0,
    steps_per_bin=10,
    bin_count=64,
    average_samples_per_bin=None,
    padding_factor=0.01,
    plot_kde=False,
    save_figure=None,
    use_axes=None,
    fit_method='kl_divergence',
    num_candidates = 10,
    tolerance = 1e-3,
    return_kde=False
):
    kde = imbal.kde.fit_kde(
        labels,
        fit_method=fit_method,
        bin_count=bin_count,
        steps_per_bin=steps_per_bin,
        average_samples_per_bin=average_samples_per_bin,
        padding_factor=padding_factor,
        num_candidates=num_candidates,
        tolerance=tolerance,
    )

    weights, approx = imbal.get_sample_densities(
        labels,
        bandwidth=kde,
        atol=atol,
        interpolation_method=interpolation_method,
        return_interpolation_samples=True,
        interpolation_samples=interpolation_samples,
        padding_factor=padding_factor
    )


    if plot_kde or use_axes or save_figure:
        imbal.kde.plot_kde_1d(
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
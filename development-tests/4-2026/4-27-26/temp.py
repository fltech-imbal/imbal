
    # Assuming data has already been loaded, and
    # model structure has been previously defined

    model = imbal.regression.Model(
        inputs=input_layer,
        outputs=output_layer
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=2e-4),
        loss='binary_crossentropy'
    )

    bandwidth = imbal.regression.fit_kde(y_train)
    densities = imbal.regression.get_sample_densities(
        y_train,
        bandwidth
    )

    model.balanced_fit(
        x_train,
        y_train,
        sample_density=densities,
        epochs=200,
        batch_size=64
    )







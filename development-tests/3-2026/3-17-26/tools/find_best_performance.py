import keras.callbacks

def find_best_performance(
    model,
    model_fit_function,
    sample_weight_candidates,
    early_stopping_parameters,
    additional_fit_parameters,
    performance_metric='val_loss',
    maximizing=False
):
    best_performance_metric = None
    best_weights = None

    initial_model_weights = model.get_weights()
    optimizer_config = model.optimizer.get_config()
    optimizer_class = type(model.optimizer)

    for weights in sample_weight_candidates:
        history = model_fit_function(
            sample_weight=weights,
            **additional_fit_parameters,
            callbacks=keras.callbacks.EarlyStopping(**early_stopping_parameters)
        )
        final_performance_metric = history.history[performance_metric][-1]
        if (best_performance_metric is None or
            (final_performance_metric < best_performance_metric and not maximizing) or
            (final_performance_metric > best_performance_metric and maximizing)):
            best_performance_metric = final_performance_metric
            best_weights = weights
        model.set_weights(initial_model_weights)
        model.optimizer = optimizer_class.from_config(optimizer_config)

    return best_weights, best_performance_metric

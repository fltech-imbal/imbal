"""Select the expert sequence on held-out folds, then fit the final models."""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold
from tensorflow import keras

from sep_c_metrics import compare_second_expert, print_validation_comparison
from sep_c_models import (
    SequentialEnsemble, build_expert, build_router, predict_in_batches, router_inputs,
)


def error_weights(targets, predictions):
    """Give every row its previous squared error divided by the mean squared error."""

    squared_errors = np.square(targets.reshape(-1).astype("float64") - predictions.reshape(-1))
    mean_error = squared_errors.mean()
    if mean_error == 0:
        return np.ones(len(targets), dtype="float32")

    return (squared_errors / mean_error).astype("float32")


def expert_weights(targets, previous_predictions):
    if previous_predictions.shape[1] == 0:
        return None

    return error_weights(targets, previous_predictions.mean(axis=1))


def router_targets(targets, expert_predictions, prefix_size):
    """Continue if any strictly longer prefix improves this row's squared error."""

    prefix_means = np.column_stack([
        expert_predictions[:, :size].mean(axis=1)
        for size in range(1, expert_predictions.shape[1] + 1)
    ])
    squared_errors = np.square(targets.reshape(-1, 1).astype("float64") - prefix_means)
    best_future_error = squared_errors[:, prefix_size:].min(axis=1)
    return (best_future_error < squared_errors[:, prefix_size - 1]).astype("float32")


def router_weights(labels, settings):
    if not settings.balance_router_classes:
        return None

    counts = np.bincount(labels.astype(int), minlength=2)
    if min(counts) == 0:
        return None

    return (len(labels) / (2 * counts[labels.astype(int)])).astype("float32")


@dataclass
class ValidationFold:
    training_indices: np.ndarray
    validation_indices: np.ndarray
    # Each column comes from an expert fitted only on this fold's training rows.
    predictions: np.ndarray


def fit_for_validation(model, features, targets, weights, validation_data, settings):
    stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=settings.early_stopping_patience,
        mode="min", restore_best_weights=True,
    )
    history = model.fit(
        features, targets, sample_weight=weights, validation_data=validation_data,
        epochs=settings.maximum_epochs, batch_size=settings.training_batch_size,
        callbacks=[stopping], shuffle=False, verbose=0,
    )
    losses = np.asarray(history.history["val_loss"])
    if not np.isfinite(losses).all():
        raise ValueError(f"Nonfinite validation loss while fitting {model.name}.")

    return int(np.argmin(losses) + 1)


def select_experts(features, targets, folds, seed, settings):
    selected_epochs = []
    validation_history = []
    previous_mse = None
    first_validation_predictions = None

    for expert_number in range(1, settings.maximum_experts + 1):
        fold_epochs = []
        candidate_predictions = []
        validation_predictions = np.empty(len(features), dtype="float64")

        for fold_number, fold in enumerate(folds, start=1):
            training = fold.training_indices
            validation = fold.validation_indices
            weights = expert_weights(targets[training], fold.predictions[training])

            keras.utils.set_random_seed(seed + expert_number * (len(folds) + 1) + fold_number)
            expert = build_expert(expert_number, features.shape[1], settings)
            print(f"Expert {expert_number}: epoch-selection fold {fold_number}/{len(folds)}")
            epoch_count = fit_for_validation(
                expert, features[training], targets[training], weights,
                (features[validation], targets[validation]), settings,
            )
            fold_epochs.append(epoch_count)
            predictions = predict_in_batches(expert, features, settings.prediction_batch_size)
            candidate = np.column_stack((fold.predictions, predictions))
            candidate_predictions.append(candidate)
            validation_predictions[validation] = candidate[validation].mean(axis=1)

        validation_mse = float(np.mean(np.square(targets.ravel() - validation_predictions)))
        if not np.isfinite(validation_mse):
            raise ValueError(f"Nonfinite validation MSE for expert {expert_number}.")

        # A perfect previous prefix has no possible relative improvement.
        accepted = previous_mse is None or (
            previous_mse > 0
            and validation_mse <= previous_mse * (1 - settings.minimum_relative_improvement)
        )
        validation_history.append({"validation_mse": validation_mse, "accepted": accepted})
        if expert_number == 1:
            first_validation_predictions = validation_predictions
        elif expert_number == 2:
            comparison = compare_second_expert(targets, first_validation_predictions, validation_predictions)
            validation_history[-1]["held_out_comparison"] = comparison
            print_validation_comparison(comparison)

        print(f"Expert {expert_number}: validation ensemble MSE {validation_mse:.6f}, "
              f"accepted={accepted}")
        if not accepted:
            break

        selected_epochs.append(int(np.round(np.mean(fold_epochs))))
        previous_mse = validation_mse
        for fold, predictions in zip(folds, candidate_predictions):
            fold.predictions = predictions

    return selected_epochs, validation_history


def select_router_epochs(features, targets, folds, router_number, seed, settings):
    fold_epochs = []
    for fold_number, fold in enumerate(folds, start=1):
        training = fold.training_indices
        validation = fold.validation_indices
        inputs = router_inputs(features, fold.predictions[:, :router_number])
        labels = router_targets(targets, fold.predictions, router_number)

        keras.utils.set_random_seed(seed + router_number * (len(folds) + 1) + fold_number)
        router = build_router(router_number, inputs.shape[1], settings)
        print(f"Router {router_number}: epoch-selection fold {fold_number}/{len(folds)}")
        epoch_count = fit_for_validation(
            router, inputs[training], labels[training], router_weights(labels[training], settings),
            (inputs[validation], labels[validation]), settings,
        )
        fold_epochs.append(epoch_count)

    return int(np.round(np.mean(fold_epochs)))


def train_ensemble(features, targets, seed, settings):
    splitter = KFold(n_splits=settings.fold_count, shuffle=True, random_state=seed)
    folds = [
        ValidationFold(training, validation, np.empty((len(features), 0), dtype="float32"))
        for training, validation in splitter.split(features)
    ]
    expert_epochs, validation_history = select_experts(features, targets, folds, seed, settings)

    # Refit the accepted sequence on all rows, recalculating weights from these
    # full-data experts. The fold models served only to select depth and epochs.
    experts = []
    predictions = np.empty((len(features), 0), dtype="float32")
    for expert_number, epoch_count in enumerate(expert_epochs, start=1):
        keras.utils.set_random_seed(seed + 1000 * expert_number)
        expert = build_expert(expert_number, features.shape[1], settings)
        print(f"Expert {expert_number}: fitting all training rows for {epoch_count} epochs")
        expert.fit(
            features, targets, sample_weight=expert_weights(targets, predictions),
            epochs=epoch_count, batch_size=settings.training_batch_size, shuffle=False, verbose=0,
        )
        expert.trainable = False
        experts.append(expert)
        next_predictions = predict_in_batches(expert, features, settings.prediction_batch_size)
        predictions = np.column_stack((predictions, next_predictions))

    routers = []
    router_epochs = []
    for router_number in range(1, len(experts)):
        epoch_count = select_router_epochs(features, targets, folds, router_number, seed, settings)
        inputs = router_inputs(features, predictions[:, :router_number])
        labels = router_targets(targets, predictions, router_number)

        keras.utils.set_random_seed(seed + 1000 * router_number)
        router = build_router(router_number, inputs.shape[1], settings)
        print(f"Router {router_number}: fitting all training rows for {epoch_count} epochs")
        router.fit(
            inputs, labels, sample_weight=router_weights(labels, settings),
            epochs=epoch_count, batch_size=settings.training_batch_size, shuffle=False, verbose=0,
        )
        routers.append(router)
        router_epochs.append(epoch_count)

    ensemble = SequentialEnsemble(
        experts, routers, settings.router_threshold, settings.prediction_batch_size
    )
    return ensemble, {
        "expert_epochs": expert_epochs, "router_epochs": router_epochs,
        "validation_history": validation_history,
    }

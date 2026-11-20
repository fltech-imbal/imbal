"""Select the expert sequence on held-out folds, then fit the final models."""

from dataclasses import dataclass
from functools import partial

import numpy as np
from sklearn.model_selection import KFold
from tensorflow import keras

from sep_c_metrics import RARE_THRESHOLD, compare_second_expert, print_validation_comparison, regression_metrics
from sep_c_models import (
    SequentialEnsemble, build_expert, build_router, predict_in_batches, prefix_prediction, router_inputs,
)

from sequential_rarity import CandidateFit, rarity_grid, tune_first_expert


def error_weights(targets, predictions):
    """Give every row its previous squared error divided by the mean squared error."""

    squared_errors = np.square(targets.reshape(-1).astype("float64") - predictions.reshape(-1))
    mean_error = squared_errors.mean()
    if mean_error == 0:
        return np.ones(len(targets), dtype="float32")

    return (squared_errors / mean_error).astype("float32")


def expert_targets(targets, previous_predictions, settings):
    if settings.residual and previous_predictions.shape[1]:
        return targets.reshape(-1, 1) - prefix_prediction(previous_predictions, residual=True)
    return targets


def expert_weights(targets, previous_predictions, settings, rarity=None):
    if settings.expert_mode == "residual_ones":
        return np.ones(len(targets), dtype="float32")
    if previous_predictions.shape[1] == 0:
        return rarity

    errors = error_weights(targets, prefix_prediction(previous_predictions, settings.residual))
    return errors if rarity is None else rarity * errors


def router_targets(targets, expert_predictions, prefix_size, residual=False):
    """Continue if any strictly longer prefix improves this row's squared error."""

    prefixes = np.column_stack([
        prefix_prediction(expert_predictions[:, :size], residual)
        for size in range(1, expert_predictions.shape[1] + 1)
    ])
    squared_errors = np.square(targets.reshape(-1, 1).astype("float64") - prefixes)
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
    rarity: np.ndarray | None = None  # Aligned with training_indices.


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


def fit_expert_candidate(features, targets, folds, seed, settings, expert_number):
    fold_epochs = []
    candidate_predictions = []
    validation_predictions = np.empty(len(features), dtype="float64")
    for fold_number, fold in enumerate(folds, start=1):
        training = fold.training_indices
        validation = fold.validation_indices
        labels = expert_targets(targets, fold.predictions, settings)
        weights = expert_weights(targets[training], fold.predictions[training], settings, fold.rarity)

        keras.utils.set_random_seed(seed + expert_number * (len(folds) + 1) + fold_number)
        expert = build_expert(expert_number, features.shape[1], settings)
        print(f"Expert {expert_number}: epoch-selection fold {fold_number}/{len(folds)}")
        epoch_count = fit_for_validation(
            expert, features[training], labels[training], weights,
            (features[validation], labels[validation]), settings,
        )
        fold_epochs.append(epoch_count)
        predictions = predict_in_batches(expert, features, settings.prediction_batch_size)
        candidate = np.column_stack((fold.predictions, predictions))
        candidate_predictions.append(candidate)
        validation_predictions[validation] = prefix_prediction(candidate[validation], settings.residual).ravel()
    return CandidateFit(int(np.round(np.mean(fold_epochs))), candidate_predictions, validation_predictions)


def select_experts(features, targets, folds, seed, settings):
    selected_epochs = []
    validation_history = []
    previous_score = None
    first_validation_predictions = None
    fit_candidate = partial(fit_expert_candidate, features, targets, folds, seed, settings)

    for expert_number in range(1, settings.maximum_experts + 1):
        step = {}
        if expert_number == 1 and settings.rarity_weighted:
            candidate, step["rarity_search"] = tune_first_expert(
                targets, folds, settings, RARE_THRESHOLD, fit_candidate, regression_metrics,
            )
        else:
            candidate = fit_candidate(expert_number)
        validation_predictions = candidate.validation_predictions
        metrics = regression_metrics(targets, validation_predictions)
        score_name = "aore" if settings.rarity_weighted else "overall_mse"
        score = metrics[score_name]
        if score is None or not np.isfinite(score):
            raise ValueError(f"Invalid validation {score_name} for expert {expert_number}.")
        step["validation_mse"] = metrics["overall_mse"]
        if settings.rarity_weighted:
            step["validation_aore"] = score

        # A perfect previous prefix has no possible relative improvement.
        meets_growth_threshold = previous_score is None or (
            previous_score > 0
            and score <= previous_score * (1 - settings.minimum_relative_improvement)
        )
        accepted = meets_growth_threshold
        if settings.residual and not settings.rarity_weighted and previous_score is not None:
            accepted = score <= previous_score
        step["accepted"] = accepted
        validation_history.append(step)
        if expert_number == 1:
            first_validation_predictions = validation_predictions
        elif expert_number == 2:
            comparison = compare_second_expert(
                targets, first_validation_predictions, validation_predictions, settings.residual
            )
            validation_history[-1]["held_out_comparison"] = comparison
            print_validation_comparison(comparison)

        print(f"Expert {expert_number}: validation ensemble {score_name} {score:.6f}, "
              f"accepted={accepted}")
        if not accepted:
            break

        selected_epochs.append(candidate.epochs)
        previous_score = score
        for fold, predictions in zip(folds, candidate.fold_predictions):
            fold.predictions = predictions
        if settings.rarity_weighted and score == 0:
            break
        if not meets_growth_threshold:
            print(f"Keeping expert {expert_number} and stopping: validation MSE improvement "
                  f"was below {settings.minimum_relative_improvement:.1%}.")
            break

    return selected_epochs, validation_history


def router_fit_weights(labels, targets, prefix, settings, rarity):
    if rarity is None:
        return router_weights(labels, settings)
    return expert_weights(targets, prefix, settings, rarity)


def select_router_epochs(features, targets, folds, router_number, seed, settings):
    fold_epochs = []
    for fold_number, fold in enumerate(folds, start=1):
        training = fold.training_indices
        validation = fold.validation_indices
        prefix = fold.predictions[:, :router_number]
        inputs = router_inputs(features, prefix, settings.residual)
        labels = router_targets(targets, fold.predictions, router_number, settings.residual)

        keras.utils.set_random_seed(seed + router_number * (len(folds) + 1) + fold_number)
        router = build_router(router_number, inputs.shape[1], settings)
        print(f"Router {router_number}: epoch-selection fold {fold_number}/{len(folds)}")
        epoch_count = fit_for_validation(
            router, inputs[training], labels[training],
            router_fit_weights(labels[training], targets[training], prefix[training], settings, fold.rarity),
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
    rarity = None
    if settings.rarity_weighted:
        selected = validation_history[0]["rarity_search"]["selected_value"]
        rarity = rarity_grid(targets, settings.expert_mode, [selected], RARE_THRESHOLD)[selected]

    # Refit the accepted sequence on all rows, recalculating weights from these
    # full-data experts. The fold models served only to select depth and epochs.
    experts = []
    predictions = np.empty((len(features), 0), dtype="float32")
    for expert_number, epoch_count in enumerate(expert_epochs, start=1):
        keras.utils.set_random_seed(seed + 1000 * expert_number)
        expert = build_expert(expert_number, features.shape[1], settings)
        print(f"Expert {expert_number}: fitting all training rows for {epoch_count} epochs")
        expert.fit(
            features, expert_targets(targets, predictions, settings),
            sample_weight=expert_weights(targets, predictions, settings, rarity),
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
        inputs = router_inputs(features, predictions[:, :router_number], settings.residual)
        labels = router_targets(targets, predictions, router_number, settings.residual)

        keras.utils.set_random_seed(seed + 1000 * router_number)
        router = build_router(router_number, inputs.shape[1], settings)
        print(f"Router {router_number}: fitting all training rows for {epoch_count} epochs")
        router.fit(
            inputs, labels, sample_weight=router_fit_weights(
                labels, targets, predictions[:, :router_number], settings, rarity
            ),
            epochs=epoch_count, batch_size=settings.training_batch_size, shuffle=False, verbose=0,
        )
        routers.append(router)
        router_epochs.append(epoch_count)

    ensemble = SequentialEnsemble(
        experts, routers, settings.router_threshold, settings.prediction_batch_size, settings.residual
    )
    return ensemble, {
        "expert_epochs": expert_epochs, "router_epochs": router_epochs,
        "validation_history": validation_history,
    }

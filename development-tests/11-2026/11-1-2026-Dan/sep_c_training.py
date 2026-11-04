"""The October training experiments, with reusable fixed-expert computations."""

import gc
import os
import random
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow import keras

from sep_c_models import (
    DENSITY_EPSILON,
    NormalizedGaussianKDE,
    build_cached_gate_trainer,
    build_expert,
    build_gate,
    gaussian_kde_density,
    predict_in_batches,
)


def set_global_determinism(seed):
    """Retain the October seeds for initialization and fold construction."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


@dataclass
class ExpertPredictionCache:
    """Four float32 values per row, rather than a row-by-reference matrix."""

    predictions: np.ndarray
    density_quality: np.ndarray

    def training_inputs(self, features, indices, settings):
        inputs = {
            "features": features[indices],
            "cached_predictions": self.predictions[indices],
        }

        if settings.density_during_gate_training:
            inputs["cached_quality"] = self.density_quality[indices]

        return inputs


def create_prediction_cache(common_expert, rare_expert, features, profiles, settings):
    """Evaluate the fixed experts once; share results across folds and reports."""

    prediction_columns = []
    quality_columns = []

    for expert_index, expert in enumerate((common_expert, rare_expert)):
        predictions = predict_in_batches(
            expert, features, settings.prediction_batch_size
        )

        prediction_columns.append(predictions.reshape(-1))

        if not settings.density_at_inference:
            quality_columns.append(np.ones(len(features), dtype="float32"))
            continue

        profile = profiles[expert_index]

        # Classification callbacks historically use NumPy's float64 KDE before
        # casting quality to float32. Regression gates use the TensorFlow layer.
        if settings.regression_gate:
            density_layer = NormalizedGaussianKDE(
                **profile,
                query_chunk_size=settings.density_query_chunk_size,
                reference_chunk_size=settings.density_reference_chunk_size,
            )

            relative_density = np.asarray(density_layer(predictions)).reshape(-1)
        else:
            densities = gaussian_kde_density(
                predictions,
                profile["training_targets"],
                profile["bandwidth"],
                settings.density_query_chunk_size,
                settings.density_reference_chunk_size,
            )

            relative_density = np.clip(
                densities / profile["maximum_density"], DENSITY_EPSILON, 1.0
            ).astype("float32")

        quality_columns.append(relative_density)

    return ExpertPredictionCache(
        predictions=np.column_stack(prediction_columns),
        density_quality=np.column_stack(quality_columns),
    )


def mix_cached_predictions(gate_trust, cache, use_density, normalize_trust=False):
    """Return the final prediction and its two interpretable expert weights."""

    if use_density:
        mixture_weights = gate_trust * cache.density_quality
    else:
        mixture_weights = np.array(gate_trust, copy=True)

    # Softmax-only mixtures already sum to one. Avoid adding an unnecessary
    # normalization there, preserving the earlier quality-free gate calculation.
    if use_density or normalize_trust:
        weight_totals = np.sum(mixture_weights, axis=1, keepdims=True)
        mixture_weights /= np.maximum(weight_totals, DENSITY_EPSILON)

    predictions = np.sum(cache.predictions * mixture_weights, axis=1)
    return predictions, mixture_weights


def aore_components(targets, predictions, rare_mask):
    target_values = np.asarray(targets).reshape(-1)
    prediction_values = np.asarray(predictions).reshape(-1)
    rare_mask = np.asarray(rare_mask).reshape(-1).astype(bool)

    if not np.any(rare_mask):
        raise ValueError("AORE requires at least one rare evaluation sample.")

    # AORE averages overall MAE and rare-region MAE, not common and rare MAE.
    # This distinction is retained in every search callback and final report.
    overall_error = mean_absolute_error(target_values, prediction_values)
    rare_error = mean_absolute_error(
        target_values[rare_mask], prediction_values[rare_mask]
    )

    return float(overall_error), float(rare_error), float((overall_error + rare_error) / 2)


def compute_sample_densities(targets):
    # Import the existing weighting implementation only when training needs it.
    # Inference from saved models does not need imbal's explanation dependencies.
    import imbal

    target_values = np.asarray(targets).reshape(-1)
    bandwidth = imbal.regression.fit_kde(target_values)
    return imbal.regression.get_sample_densities(target_values, bandwidth)


def reciprocal_weights(densities, alpha):
    import imbal

    # Keep the existing importance mapping and normalization exactly as defined
    # by imbal. Only the expensive density inputs are reused across candidates.
    weights = imbal.regression.reciprocal_importance(densities, alpha=float(alpha))
    return np.asarray(weights, dtype="float32").reshape(-1)


def pack_expert_targets(targets, weights=None):
    target_values = np.asarray(targets, dtype="float32").reshape(-1)

    if weights is None:
        weights = np.ones(len(target_values), dtype="float32")

    weights = np.asarray(weights, dtype="float32").reshape(-1)

    if len(weights) != len(target_values) or not np.all(np.isfinite(weights)):
        raise ValueError("Each target needs one finite importance weight.")

    if np.any(weights < 0) or np.sum(weights) <= 0:
        raise ValueError("Importance weights must be nonnegative with positive total.")

    return np.column_stack((target_values, weights))


def subbin_average_error(targets, predictions, midpoint):
    target_values = np.asarray(targets).reshape(-1)
    prediction_values = np.asarray(predictions).reshape(-1)
    lower_subbin = target_values <= midpoint
    upper_subbin = ~lower_subbin

    if not np.any(lower_subbin) or not np.any(upper_subbin):
        return np.inf

    lower_error = mean_absolute_error(
        target_values[lower_subbin], prediction_values[lower_subbin]
    )
    upper_error = mean_absolute_error(
        target_values[upper_subbin], prediction_values[upper_subbin]
    )

    return float((lower_error + upper_error) / 2)


class ExpertSubbinEarlyStopping(keras.callbacks.Callback):
    """Select expert epochs using equal weight for the two target sub-bins."""

    def __init__(self, features, targets, midpoint, settings):
        super().__init__()
        self.features = features
        self.targets = targets
        self.midpoint = midpoint
        self.settings = settings

        self.best_error = np.inf
        self.best_epoch = 1
        self.best_weights = None
        self.epochs_without_improvement = 0

    def on_epoch_end(self, epoch, logs=None):
        predictions = predict_in_batches(
            self.model, self.features, self.settings.prediction_batch_size
        )

        validation_error = subbin_average_error(
            self.targets, predictions, self.midpoint
        )

        if validation_error < self.best_error:
            self.best_error = validation_error
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.settings.early_stopping_patience:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


class GateAOREEarlyStopping(keras.callbacks.Callback):
    """Measure the same validation mixture without rerunning frozen experts."""

    def __init__(self, gate, features, targets, region_labels, cache, settings):
        super().__init__()
        self.gate = gate
        self.features = features
        self.targets = targets
        self.region_labels = region_labels
        self.cache = cache
        self.settings = settings

        self.best_aore = np.inf
        self.best_epoch = 1
        self.best_weights = None
        self.epochs_without_improvement = 0

    def on_epoch_end(self, epoch, logs=None):
        trust = predict_in_batches(
            self.gate, self.features, self.settings.prediction_batch_size
        )

        # The no-quality regression experiment also excludes quality from its
        # early-stopping metric. Density is added only for final inference there.
        use_density = (
            self.settings.density_during_gate_training
            if self.settings.regression_gate
            else self.settings.density_at_inference
        )

        predictions, _ = mix_cached_predictions(
            trust,
            self.cache,
            use_density=use_density,
            normalize_trust=self.settings.sigmoid_gate,
        )

        _, _, validation_aore = aore_components(
            self.targets, predictions, self.region_labels
        )

        if logs is not None:
            logs["val_ensemble_aore"] = validation_aore

        if validation_aore < self.best_aore:
            self.best_aore = validation_aore
            self.best_epoch = epoch + 1
            self.best_weights = self.gate.get_weights()
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.settings.early_stopping_patience:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.gate.set_weights(self.best_weights)


def train_classification_experiment_expert(features, targets, expert_name, settings):
    """Select epochs once for classification experts or MSE-only regression experts."""

    if len(features) < settings.fold_count:
        raise ValueError(f"{expert_name} has too few samples for the requested folds.")

    splitter = KFold(
        n_splits=settings.fold_count, shuffle=True, random_state=settings.seed
    )
    selected_epochs = []

    for fold_number, (training_indices, validation_indices) in enumerate(
        splitter.split(features), start=1
    ):
        keras.backend.clear_session()
        set_global_determinism(settings.seed + fold_number)
        expert = build_expert(expert_name, features.shape[1], settings)

        stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=settings.early_stopping_patience,
            mode="min",
            restore_best_weights=True,
        )

        print(f"{expert_name}: epoch-selection fold {fold_number}/{settings.fold_count}")
        history = expert.fit(
            features[training_indices],
            targets[training_indices],
            validation_data=(features[validation_indices], targets[validation_indices]),
            epochs=settings.maximum_epochs,
            batch_size=settings.training_batch_size,
            callbacks=[stopping],
            shuffle=False,
            verbose=0,
        )

        selected_epochs.append(int(np.argmin(history.history["val_loss"]) + 1))

        # Histories and callbacks reference their models. Release those references
        # before the next clear_session rather than carrying an old fold forward.
        del history, stopping, expert
        gc.collect()

    epoch_count = max(1, int(np.round(np.mean(selected_epochs))))
    expert_seed_offset = 1000 if expert_name == "common_expert" else 2000

    keras.backend.clear_session()
    set_global_determinism(settings.seed + expert_seed_offset)
    expert = build_expert(expert_name, features.shape[1], settings)

    expert.fit(
        features,
        targets,
        epochs=epoch_count,
        batch_size=settings.training_batch_size,
        shuffle=False,
        verbose=0,
    )

    print(f"{expert_name}: final training used {epoch_count} epochs")
    return expert


def train_regression_experiment_expert(features, targets, expert_name, settings):
    """Retain reciprocal-alpha selection using the original sub-bin metric."""

    target_values = np.asarray(targets).reshape(-1)
    midpoint = float(np.min(target_values) + (np.max(target_values) - np.min(target_values)) / 2)
    subbin_labels = (target_values > midpoint).astype("int32")
    subbin_counts = np.bincount(subbin_labels, minlength=2)

    if np.min(subbin_counts) < settings.fold_count:
        raise ValueError(f"{expert_name}: each sub-bin needs at least one row per fold.")

    relative_imbalance = abs(int(subbin_counts[0]) - int(subbin_counts[1])) / max(subbin_counts)
    use_importance_weights = relative_imbalance >= settings.subbin_balance_tolerance
    candidates = settings.alpha_candidates if use_importance_weights else (None,)

    expert_seed_offset = 1000 if expert_name == "common_expert" else 2000
    selection_seed = settings.seed + expert_seed_offset
    splitter = StratifiedKFold(
        n_splits=settings.fold_count, shuffle=True, random_state=selection_seed
    )

    print(f"{expert_name}: sub-bin counts {subbin_counts.tolist()}, midpoint {midpoint:.6f}")
    folds = []

    # Fit each fold's weighting density only on that fold's training targets.
    # Reuse these densities for all alpha values, without changing the estimator.
    for training_indices, validation_indices in splitter.split(features, subbin_labels):
        densities = (
            compute_sample_densities(targets[training_indices])
            if use_importance_weights else None
        )
        folds.append((training_indices, validation_indices, densities))

    best_error = np.inf
    best_alpha = None
    best_epoch_count = None

    for alpha in candidates:
        fold_errors = []
        fold_epochs = []

        for fold_number, (training_indices, validation_indices, densities) in enumerate(
            folds, start=1
        ):
            keras.backend.clear_session()
            set_global_determinism(selection_seed + fold_number)
            expert = build_expert(expert_name, features.shape[1], settings)

            stopping = ExpertSubbinEarlyStopping(
                features[validation_indices], targets[validation_indices], midpoint, settings
            )

            weights = reciprocal_weights(densities, alpha) if use_importance_weights else None
            packed_targets = pack_expert_targets(targets[training_indices], weights)

            print(f"{expert_name}: alpha={alpha}, fold {fold_number}/{settings.fold_count}")
            expert.fit(
                features[training_indices],
                packed_targets,
                epochs=settings.maximum_epochs,
                batch_size=settings.training_batch_size,
                callbacks=[stopping],
                shuffle=False,
                verbose=0,
            )

            fold_errors.append(stopping.best_error)
            fold_epochs.append(stopping.best_epoch)
            del expert, stopping, packed_targets
            gc.collect()

        mean_error = float(np.mean(fold_errors))
        epoch_count = max(1, int(np.round(np.mean(fold_epochs))))
        print(f"  alpha={alpha}: mean sub-bin error={mean_error:.6f}, epochs={epoch_count}")

        if mean_error < best_error:
            best_error = mean_error
            best_alpha = alpha
            best_epoch_count = epoch_count

    if best_epoch_count is None:
        raise ValueError("No finite expert validation result was found.")

    # The final expert uses all targets in its own region, just as in October.
    # Its density is computed once after selecting alpha, not once per epoch.
    final_weights = None

    if use_importance_weights:
        final_weights = reciprocal_weights(compute_sample_densities(targets), best_alpha)

    keras.backend.clear_session()
    set_global_determinism(selection_seed + 1000)
    expert = build_expert(expert_name, features.shape[1], settings)

    expert.fit(
        features,
        pack_expert_targets(targets, final_weights),
        epochs=best_epoch_count,
        batch_size=settings.training_batch_size,
        shuffle=False,
        verbose=0,
    )

    print(f"{expert_name}: selected alpha={best_alpha}, epochs={best_epoch_count}")
    return expert


def train_experts(features, targets, region_labels, settings):
    # MSE-only experiments reuse the classification method's single KFold pass.
    # This changes expert fitting only; gate candidate selection stays independent.
    train_one_expert = (
        train_regression_experiment_expert
        if settings.regression_gate and not settings.mse_only_expert_training
        else train_classification_experiment_expert
    )

    # Experts still specialize on disjoint target regions. No new samples or
    # test labels enter either the common or rare expert's training set.
    common_mask = np.asarray(region_labels).reshape(-1) == 0
    rare_mask = ~common_mask

    common_expert = train_one_expert(
        features[common_mask], targets[common_mask], "common_expert", settings
    )
    rare_expert = train_one_expert(
        features[rare_mask], targets[rare_mask], "rare_expert", settings
    )

    common_expert.trainable = False
    rare_expert.trainable = False
    return common_expert, rare_expert


def fit_gate_candidate(
    gate,
    features,
    targets,
    labels,
    cache,
    indices,
    candidate,
    epochs,
    settings,
    callbacks=None,
    validation_indices=None,
    densities=None,
):
    """Fit one gate while keeping its classification/regression objective explicit."""

    fit_arguments = {
        "epochs": epochs,
        "batch_size": settings.training_batch_size,
        "callbacks": callbacks or [],
        "shuffle": False,
        "verbose": 0,
    }

    if settings.regression_gate:
        trainer = build_cached_gate_trainer(gate, features.shape[1], settings)
        sample_weights = reciprocal_weights(densities, candidate)

        trainer.fit(
            cache.training_inputs(features, indices, settings),
            targets[indices],
            sample_weight=sample_weights,
            **fit_arguments,
        )
    else:
        # Keep classification validation_loss reporting in addition to the
        # AORE callback. Selection continues to use the ensemble's AORE.
        if validation_indices is not None:
            fit_arguments["validation_data"] = (
                features[validation_indices], labels[validation_indices]
            )

        gate.fit(
            features[indices],
            labels[indices],
            class_weight={0: 1.0, 1: float(candidate)},
            **fit_arguments,
        )


def select_and_train_gate(features, targets, region_labels, cache, settings):
    labels = np.asarray(region_labels).reshape(-1).astype("int32")

    if np.min(np.bincount(labels, minlength=2)) < settings.fold_count:
        raise ValueError("Each region needs at least one validation row per fold.")

    splitter = StratifiedKFold(
        n_splits=settings.fold_count, shuffle=True, random_state=settings.seed
    )
    folds = []

    for training_indices, validation_indices in splitter.split(features, labels):
        densities = (
            compute_sample_densities(targets[training_indices])
            if settings.regression_gate else None
        )
        folds.append((training_indices, validation_indices, densities))

    candidates = (
        settings.alpha_candidates
        if settings.regression_gate else settings.rare_class_weight_candidates
    )

    best_aore = np.inf
    best_candidate = None
    best_epoch_count = None

    for candidate in candidates:
        fold_aores = []
        fold_epochs = []

        for fold_number, (training_indices, validation_indices, densities) in enumerate(
            folds, start=1
        ):
            keras.backend.clear_session()
            set_global_determinism(settings.seed + 100 + fold_number)
            gate = build_gate(features.shape[1], settings)

            validation_cache = ExpertPredictionCache(
                cache.predictions[validation_indices],
                cache.density_quality[validation_indices],
            )

            stopping = GateAOREEarlyStopping(
                gate,
                features[validation_indices],
                targets[validation_indices],
                labels[validation_indices],
                validation_cache,
                settings,
            )

            print(f"Gate: candidate={candidate}, fold {fold_number}/{settings.fold_count}")
            fit_gate_candidate(
                gate, features, targets, labels, cache, training_indices,
                candidate, settings.maximum_epochs, settings,
                callbacks=[stopping], validation_indices=validation_indices,
                densities=densities,
            )

            fold_aores.append(stopping.best_aore)
            fold_epochs.append(stopping.best_epoch)
            del gate, stopping, validation_cache
            gc.collect()

        mean_aore = float(np.mean(fold_aores))
        epoch_count = max(1, int(np.round(np.mean(fold_epochs))))
        print(f"  candidate={candidate}: mean AORE={mean_aore:.6f}, epochs={epoch_count}")

        if mean_aore < best_aore:
            best_aore = mean_aore
            best_candidate = candidate
            best_epoch_count = epoch_count

    if best_epoch_count is None:
        raise ValueError("No finite gate validation result was found.")

    keras.backend.clear_session()
    set_global_determinism(settings.seed + 3000)
    gate = build_gate(features.shape[1], settings)

    final_densities = compute_sample_densities(targets) if settings.regression_gate else None
    fit_gate_candidate(
        gate, features, targets, labels, cache, slice(None),
        best_candidate, best_epoch_count, settings, densities=final_densities,
    )

    print(f"Gate: selected candidate={best_candidate}, epochs={best_epoch_count}")
    return gate

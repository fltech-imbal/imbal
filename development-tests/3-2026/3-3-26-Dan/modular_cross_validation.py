from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Protocol, List
import numpy as np
import warnings

import imbal
import imbal.util.backend as backend
from tensorflow import keras
from tensorflow.keras import layers
from stratified_split_for_k_folds import stratified_kfold_indices
from decoupled_fit_with_cross_val import decoupled_fit_with_cross_val_stage_one
from decoupled_fit_with_cross_val import decoupled_fit_with_cross_val_stage_two
from dense_weight import denseweight_sample_weights


DEFAULT_MAX_EPOCHS = 100_000


# ---------- Types ----------
ValidationData = Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]

class SupportsFit(Protocol):
    def fit(self, *args, **kwargs): ...


# ---------- EarlyStopping cloning (optional, but matches your current pattern) ----------
def clone_early_stopping(cb):
    # Works for keras.callbacks.EarlyStopping and tf.keras.callbacks.EarlyStopping
    return cb.__class__(
        monitor=getattr(cb, "monitor", "val_loss"),
        min_delta=getattr(cb, "min_delta", 0.0),
        patience=getattr(cb, "patience", 0),
        verbose=getattr(cb, "verbose", 0),
        mode=getattr(cb, "mode", "auto"),
        baseline=getattr(cb, "baseline", None),
        restore_best_weights=getattr(cb, "restore_best_weights", False),
        start_from_epoch=getattr(cb, "start_from_epoch", 0),  # newer keras
    )


def clone_convergence_stopping(cb):
    return cb.__class__(
        monitor=getattr(cb, "monitor", "val_loss"),
        tol=getattr(cb, "tol", 1e-4),
        patience=getattr(cb, "patience", 5),
        restore_best_weights=getattr(cb, "restore_best_weights", True),
    )


def normalize_sample_weights(sw, eps: float = 1e-8) -> np.ndarray:
    """
    Rescale sample weights so that sum(weights) == number of samples.
    This matches Keras' implicit expectation.
    """
    sw = np.asarray(sw, dtype=np.float32)
    n = sw.shape[0]
    s = np.sum(sw)

    if not np.isfinite(s) or s < eps:
        raise ValueError("Invalid sample weights: sum is zero or non-finite.")

    if not np.isclose(s, n):
        sw = sw * (n / s)

    return sw


# ---------- Params objects ----------
@dataclass
class BaseFitParams:
    """
    Common union container. Subclasses can add/override fields as needed.
    """
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None

    sample_weight: Optional[np.ndarray] = None
    validation_data: Optional[ValidationData] = None
    validation_split: Optional[float] = None

    batch_size: int = 32
    shuffle: bool = True

    callbacks: Optional[list] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def with_fold_data(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray,
        y_va: np.ndarray,
        sw_tr: Optional[np.ndarray],
        sw_va: Optional[np.ndarray],
        callbacks: Optional[list],
    ) -> BaseFitParams:
        """
        Return a copy configured for a fold.
        Subclasses can override to customize behavior.
        """
        p = self.copy()
        p.x = X_tr
        p.y = y_tr
        p.sample_weight = sw_tr
        p.validation_data = (X_va, y_va, sw_va) if sw_va is not None else (X_va, y_va)
        p.callbacks = callbacks
        return p

    def copy(self):
        # dataclasses.replace is fine too; this keeps it explicit
        cls = self.__class__
        return cls(**{**self.__dict__, "kwargs": dict(self.kwargs)})

    def to_fit_kwargs_for_fold(self, *, max_epochs: int) -> Dict[str, Any]:
        """
        Convert params to kwargs for the fold training call.
        Subclasses override to adjust epochs handling, etc.
        """
        out = dict(self.kwargs)
        out.update(
            dict(
                x=self.x,
                y=self.y,
                sample_weight=self.sample_weight,
                validation_data=self.validation_data,
                validation_split=self.validation_split,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            )
        )
        if self.callbacks is not None:
            out["callbacks"] = self.callbacks

        # For k-fold search we generally drive epochs here:
        out["epochs"] = max_epochs
        return out

    def to_fit_kwargs_for_full_train(self, *, epochs: int) -> Dict[str, Any]:
        """
        Convert params to kwargs for the final full training call.
        """
        out = dict(self.kwargs)
        out.update(
            dict(
                x=self.x,
                y=self.y,
                sample_weight=self.sample_weight,
                validation_data=self.validation_data,
                validation_split=self.validation_split,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                epochs=epochs,
            )
        )
        # intentionally DO NOT add callbacks here
        return out


@dataclass
class RegularFitParams(BaseFitParams):
    stratify_batches: Optional[bool] = False

    def to_fit_kwargs_for_fold(self, *, max_epochs: int) -> Dict[str, Any]:
        out = super().to_fit_kwargs_for_fold(max_epochs=max_epochs)
        out["stratify_batches"] = self.stratify_batches
        return out

    def to_fit_kwargs_for_full_train(self, *, epochs: int) -> Dict[str, Any]:
        out = super().to_fit_kwargs_for_full_train(epochs=epochs)
        out["stratify_batches"] = self.stratify_batches
        return out


@dataclass
class BalancedFitParams(BaseFitParams):
    stratify_batches: Optional[bool] = False
    class_weight: Optional[dict] = None
    sample_density: Optional[np.ndarray] = None

    def to_fit_kwargs_for_fold(self, *, max_epochs: int) -> Dict[str, Any]:
        out = super().to_fit_kwargs_for_fold(max_epochs=max_epochs)
        out["stratify_batches"] = self.stratify_batches
        out["class_weight"] = self.class_weight
        out["sample_density"] = self.sample_density
        return out

    def to_fit_kwargs_for_full_train(self, *, epochs: int) -> Dict[str, Any]:
        out = super().to_fit_kwargs_for_full_train(epochs=epochs)
        out["stratify_batches"] = self.stratify_batches
        out["class_weight"] = self.class_weight
        out["sample_density"] = self.sample_density
        return out


@dataclass
class DecoupledFitStageOneParams(BaseFitParams):
    """
    Note: decoupled_fit signature includes epochs explicitly.
    We'll still set epochs from the driver (k-fold search and final train).
    """
    epochs: int = 1  # default, but overridden by driver
    stratify_batches: Optional[bool] = False

    def to_fit_kwargs_for_fold(self, *, max_epochs: int) -> Dict[str, Any]:
        out = super().to_fit_kwargs_for_fold(max_epochs=max_epochs)
        # keep compatibility with signature; driver sets epochs anyway
        out["epochs"] = max_epochs
        out["stratify_batches"] = self.stratify_batches
        return out

    def to_fit_kwargs_for_full_train(self, *, epochs: int) -> Dict[str, Any]:
        out = super().to_fit_kwargs_for_full_train(epochs=epochs)
        out["epochs"] = epochs
        out["stratify_batches"] = self.stratify_batches
        return out


@dataclass
class DecoupledFitStageTwoParams(BaseFitParams):
    """
    Note: decoupled_fit signature includes epochs explicitly.
    We'll still set epochs from the driver (k-fold search and final train).
    """
    epochs: int = 1  # default, but overridden by driver
    stratify_batches: Optional[bool] = False
    class_weight: Optional[dict] = None
    sample_density: Optional[np.ndarray] = None

    def to_fit_kwargs_for_fold(self, *, max_epochs: int) -> Dict[str, Any]:
        out = super().to_fit_kwargs_for_fold(max_epochs=max_epochs)
        # keep compatibility with signature; driver sets epochs anyway
        out["epochs"] = max_epochs
        out["stratify_batches"] = self.stratify_batches
        out["class_weight"] = self.class_weight
        out["sample_density"] = self.sample_density
        return out

    def to_fit_kwargs_for_full_train(self, *, epochs: int) -> Dict[str, Any]:
        out = super().to_fit_kwargs_for_full_train(epochs=epochs)
        out["epochs"] = epochs
        out["stratify_batches"] = self.stratify_batches
        out["class_weight"] = self.class_weight
        out["sample_density"] = self.sample_density
        return out


# ---------- Strategy (subclasses per type) ----------
class FitStrategy(Protocol):
    def make_call(self, model: Any) -> Callable[[Dict[str, Any]], Any]: ...
    def fold_prepare(
        self,
        params: BaseFitParams,
        *,
        tr_idx: np.ndarray,
        va_idx: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
        early_stop_cb: Optional[Any],
    ) -> Tuple[BaseFitParams, Optional[Any]]: ...
    def extract_val_losses(self, history: Any) -> List[float]: ...
    def model_prepare(self, model: Any) -> Any: ...
    def finalize_model_after_training(self, model: Any) -> Any: ...


@dataclass
class BaseStrategy:
    """
    Shared behavior; subclasses override only what differs.
    """

    # def _fresh_callbacks(self, early_stop_cb):
    #     if early_stop_cb is None:
    #         return None
    #     # Keras expects a list[Callback], not a nested list
    #     return [clone_early_stopping(early_stop_cb)]

    def _fresh_callbacks(self, convergence_stop_cb):
        if convergence_stop_cb is None:
            return None
        # Keras expects a list[Callback], not a nested list
        return [clone_convergence_stopping(convergence_stop_cb)]

    def fold_prepare(
        self,
        params: BaseFitParams,
        *,
        tr_idx: np.ndarray,
        va_idx: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
        early_stop_cb: Optional[Any],
    ) -> Tuple[BaseFitParams, Optional[Any]]:
        X_tr, X_va = x[tr_idx], x[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        sw_tr = normalize_sample_weights(sample_weight[tr_idx]) if sample_weight is not None else None
        sw_va = normalize_sample_weights(sample_weight[va_idx]) if sample_weight is not None else None

        callbacks = self._fresh_callbacks(early_stop_cb)
        fold_params = params.with_fold_data(X_tr, y_tr, X_va, y_va, sw_tr, sw_va, callbacks)
        return fold_params, early_stop_cb

    def extract_val_losses(self, history: Any) -> List[float]:
        # Robust to missing val_loss
        h = getattr(history, "history", {}) or {}
        return list(h.get("val_loss", []))

    def model_prepare(self, model):
        return model

    def finalize_model_after_training(self, model):
        return model


@dataclass
class RegularFitStrategy(BaseStrategy):
    def make_call(self, model: Any) -> Callable[[Dict[str, Any]], Any]:
        # lambda per suggestion
        return lambda fit_kwargs: model.fit(**fit_kwargs)


@dataclass
class BalancedFitStrategy(BaseStrategy):
    def make_call(self, model: Any) -> Callable[[Dict[str, Any]], Any]:
        return lambda fit_kwargs: model.balanced_fit(**fit_kwargs)

    # If you need special per-fold behavior (e.g. derive sample_density per fold),
    # override fold_prepare(...) here.


@dataclass
class DecoupledFitStageOneStrategy(BaseStrategy):
    def make_call(self, model: Any) -> Callable[[Dict[str, Any]], Any]:
        return lambda fit_kwargs: decoupled_fit_with_cross_val_stage_one(model, **fit_kwargs)

    # If you need special per-fold behavior for decoupled training, override fold_prepare(...).


@dataclass
class DecoupledFitStageTwoStrategy(BaseStrategy):
    def make_call(self, model: Any) -> Callable[[Dict[str, Any]], Any]:
        return lambda fit_kwargs: decoupled_fit_with_cross_val_stage_two(model, **fit_kwargs)

    def clone_layer(self, layer: keras.layers.Layer) -> keras.layers.Layer:
        # Creates a fresh layer with same settings but NEW weights / NEW build shape
        return layer.__class__.from_config(layer.get_config())

    def model_prepare(self, model: Any) -> Any:
        # --- find representation layer index (your existing logic) ---
        representation_layer_index = backend.tools.positive_model_layer_index(
            model, model._representation_layer_index
        )
        found_layer, found_index = imbal.util.get_representation_layer_index(
            model, desired_layer_index=representation_layer_index
        )
        if found_index is None:
            raise ValueError("Unable to find viable representation layer...")
        if representation_layer_index > found_index:
            warnings.warn(
                f"Overriding representation layer to layer {found_index} "
                f"(originally {representation_layer_index})"
            )
            representation_layer_index = found_index

        # Layers up through rep layer are "stage one" and will be frozen
        frozen_layers = model.layers[: representation_layer_index + 1]
        frozen_names = {l.name for l in frozen_layers}

        # --- rebuild graph: reuse old layers, insert new hardcoded layer ---
        inputs = model.inputs
        if isinstance(inputs, (list, tuple)):
            x = inputs[0]
        else:
            x = inputs

        # Skip InputLayer if present at index 0
        start_idx = 1 if isinstance(model.layers[0], keras.layers.InputLayer) else 0

        # run through trunk up to rep layer
        for layer in model.layers[start_idx : representation_layer_index + 1]:
            x = layer(x)

        # >>> HARD-CODED EXTRA LAYER <<<
        x = layers.Dense(4, activation="relu", name="stage2_extra_dense")(x)

        # clone the rest of the head so input shapes match
        for layer in model.layers[representation_layer_index + 1:]:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            new_layer = self.clone_layer(layer)
            x = new_layer(x)

        new_model = model.__class__(inputs=model.inputs, outputs=x, name=model.name + "_stage2")

        # Copy over the custom imbal fields that balanced_fit/decoupled logic expects
        for attr in [
            "_mode_enum",
            "_mode_subpackage",
            "_representation_layer_index",
            "_use_decoder_branch",
            "_generate_decoder_branch",
            "_decoder_branch",
            "_extended_model",
        ]:
            if hasattr(model, attr):
                setattr(new_model, attr, getattr(model, attr))

        # Freeze stage-one layers by name (these are shared layer objects)
        for layer in new_model.layers:
            if layer.name in frozen_names:
                layer.trainable = False

        # Re-init all trainable layers (includes new stage2 layer + the head),
        # keeping frozen trunk weights intact.
        for layer in new_model.layers:
            if layer.trainable and hasattr(layer, "kernel_initializer") and hasattr(layer, "bias_initializer"):
                layer.set_weights([
                    layer.kernel_initializer(shape=np.asarray(layer.kernel.shape)),
                    layer.bias_initializer(shape=np.asarray(layer.bias.shape)),
                ])

        # Keep decoder-branch handling consistent
        if getattr(new_model, "_use_decoder_branch", False):
            for layer in new_model._decoder_branch:
                layer.trainable = False

        print(new_model.summary())

        return new_model

    def finalize_model_after_training(self, model):
        model._use_decoder_branch = model._generate_decoder_branch

        if model._generate_decoder_branch:
            model._extended_model.trainable = True

        return model


# ---------- Main k-fold function ----------
def fit_k_folds_modular(
    model: imbal.util.backend.Model,
    x: np.ndarray,
    y: np.ndarray,
    *,
    strategy: FitStrategy,
    params: BaseFitParams,
    batch_size: int = 64,
    num_folds: int = 5,
    shuffle: bool = True,
    seed: int = 0,
    mode=None,  # pass through to stratified_kfold_indices
    min_or_max="min",
    metric="val_loss",
    class_weight_candidates: Optional[List[dict]] = None,
    alpha_candidates: Optional[List[float]] = None,
    alpha_eps: float = 1e-4,
):
    """
    Modular k-fold driver. Optionally selects best hyperparameter candidate via CV:
      - classification: class_weight_candidates=[{...}, {...}]
      - regression:     alpha_candidates=[0.0, 0.5, 1.0, ...] -> converted internally to sample_weight

    Selection rule:
      - For each candidate: score = mean over folds of min(metric)
      - Choose candidate with lowest score
      - Then full-train once on all training data with:
          * chosen candidate hyperparameter
          * epochs = avg-best-epoch computed *within that candidate's CV run*
    """

    if class_weight_candidates is not None and alpha_candidates is not None:
        raise ValueError("Pass only one of class_weight_candidates or alpha_candidates, not both.")

    # Pull early stopping from params.callbacks (if present)
    callbacks = params.callbacks or []
    early_stop = callbacks[0] if callbacks else None

    # Driver overrides batch_size if you want it centralized
    base_params = params.copy()
    base_params.batch_size = batch_size

    compile_cfg = model.get_compile_config()
    model = strategy.model_prepare(model)
    initial_weights = model.get_weights()

    # Fit callable (lambda)
    fit_call = strategy.make_call(model)

    # Helper: build candidate param variants (or just a single baseline run)
    allowed_for_sweep = isinstance(strategy, (BalancedFitStrategy, DecoupledFitStageTwoStrategy))

    candidates: List[Tuple[str, BaseFitParams, dict]] = []

    if class_weight_candidates is not None:
        if not allowed_for_sweep:
            raise ValueError(
                "class_weight_candidates is only supported for BalancedFitStrategy and DecoupledFitStageTwoStrategy."
            )
        for i, cw in enumerate(class_weight_candidates):
            p = base_params.copy()
            if not hasattr(p, "class_weight"):
                raise ValueError(
                    "params does not support class_weight; only BalancedFitParams/DecoupledFitStageTwoParams do."
                )
            setattr(p, "class_weight", cw)
            candidates.append((f"class_weight[{i}]", p, {"class_weight": cw}))

    elif alpha_candidates is not None:
        if not allowed_for_sweep:
            raise ValueError(
                "alpha_candidates is only supported for BalancedFitStrategy and DecoupledFitStageTwoStrategy."
            )

        # Build sample densities ONCE (independent of alpha)
        labels_kde = y.reshape(-1).copy()
        kde = imbal.regression.fit_kde(labels_kde)
        densities = imbal.regression.get_sample_densities(labels_kde, kde)

        candidates: List[Tuple[str, BaseFitParams, dict]] = []

        for alpha in alpha_candidates:
            p = base_params.copy()
            sw = denseweight_sample_weights(densities, float(alpha), alpha_eps)
            p.sample_weight = sw
            candidates.append((f"alpha={float(alpha)}", p, {"alpha": float(alpha)}))

    else:
        # No sweep: single run with the provided params
        candidates.append(("baseline", base_params, {}))

    # ---- Run CV for each candidate, pick best ----
    best_name: str = candidates[0][0]
    best_params: BaseFitParams = candidates[0][1]
    best_meta: dict = candidates[0][2]
    best_cv_score: float = float("inf") if min_or_max == "min" else -float("inf")
    best_avg_best_epoch: int = 1

    # store CV summary for every candidate
    cv_by_candidate: Dict[str, dict] = {}

    for cand_name, cand_params, cand_meta in candidates:
        # Reset model to the same initial state for this candidate
        model.set_weights(initial_weights)

        fold_best_epochs: List[int] = []
        fold_best_metric_values: List[float] = []

        fold_metrics_at_best_epoch: List[Dict[str, float]] = []

        for fold, (tr_idx, va_idx) in enumerate(
                stratified_kfold_indices(y, k=num_folds, seed=seed, mode=mode, shuffle=shuffle)
        ):
            model.compile_from_config(compile_cfg)
            model.set_weights(initial_weights)
            model.reset_metrics()

            fold_params, _ = strategy.fold_prepare(
                cand_params,
                tr_idx=tr_idx,
                va_idx=va_idx,
                x=x,
                y=y,
                sample_weight=cand_params.sample_weight,
                early_stop_cb=early_stop,
            )

            fold_fit_kwargs = fold_params.to_fit_kwargs_for_fold(max_epochs=DEFAULT_MAX_EPOCHS)
            history = fit_call(fold_fit_kwargs)

            metric_values = history.history.get(metric, [])
            if not metric_values:
                continue

            best_epoch_idx = int(np.argmax(metric_values)) if min_or_max == "max" else int(np.argmin(metric_values))
            fold_best_epochs.append(best_epoch_idx + 1)
            fold_best_metric_values.append(float(metric_values[best_epoch_idx]))

            # snapshot *all* logged keys at best epoch
            snap: Dict[str, float] = {}
            for k, v in (history.history or {}).items():
                if v is None:
                    continue
                if best_epoch_idx < len(v):
                    try:
                        snap[k] = float(v[best_epoch_idx])
                    except Exception:
                        pass
            fold_metrics_at_best_epoch.append(snap)

        # Candidate summary
        avg_best_epoch = int(np.rint(np.mean(fold_best_epochs))) if fold_best_epochs else 1
        avg_best_epoch = max(1, avg_best_epoch)

        cv_score = float(np.mean(fold_best_metric_values)) if fold_best_metric_values else float("inf")

        # Average each history key across folds (using only folds that had that key)
        avg_metrics_at_best_epoch: Dict[str, float] = {}
        if fold_metrics_at_best_epoch:
            keys = set().union(*(d.keys() for d in fold_metrics_at_best_epoch))
            for k in keys:
                vals = [d[k] for d in fold_metrics_at_best_epoch if k in d]
                if vals:
                    avg_metrics_at_best_epoch[k] = float(np.mean(vals))

        print(
            f"[CV] candidate={cand_name} | mean_best_{metric}={cv_score:.6f} | avg_best_epoch={avg_best_epoch}"
        )

        cv_by_candidate[cand_name] = {
            "meta": cand_meta,
            "mean_best_metric_value": cv_score,
            "avg_best_epoch": avg_best_epoch,
            "avg_metrics_at_best_epoch": avg_metrics_at_best_epoch,
        }

        # Track the winner (unchanged behavior)
        if min_or_max == "min":
            if cv_score < best_cv_score:
                best_name = cand_name
                best_params = cand_params
                best_meta = cand_meta
                best_cv_score = cv_score
                best_avg_best_epoch = avg_best_epoch
        else:
            if cv_score > best_cv_score:
                best_name = cand_name
                best_params = cand_params
                best_meta = cand_meta
                best_cv_score = cv_score
                best_avg_best_epoch = avg_best_epoch

    print(f"Selected candidate: {best_name} (mean_best_metric_value={best_cv_score:.4f})")
    print("Best epoch (average):", best_avg_best_epoch)

    # ---- Full retrain once using the chosen candidate + its avg best epoch ----
    model.set_weights(initial_weights)
    model.compile_from_config(compile_cfg)
    model.reset_metrics()

    full_params = best_params.copy()
    full_params.x = x
    full_params.y = y
    full_fit_kwargs = full_params.to_fit_kwargs_for_full_train(epochs=best_avg_best_epoch)

    full_history = fit_call(full_fit_kwargs)
    model = strategy.finalize_model_after_training(model)

    selection_info = {
        "selected_candidate_name": best_name,
        "selected_alpha": best_meta.get("alpha"),
        "selected_class_weight": best_meta.get("class_weight"),
        "mean_best_metric_value": best_cv_score,
        "avg_best_epoch": best_avg_best_epoch,
        "cv_by_candidate": cv_by_candidate,
    }

    return full_history, model, selection_info

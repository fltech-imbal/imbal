from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Protocol, List
import numpy as np
import warnings

import imbal
import imbal.util.backend as backend
from stratified_split_for_k_folds import stratified_kfold_indices
from decoupled_fit_with_cross_val import decoupled_fit_with_cross_val_stage_one
from decoupled_fit_with_cross_val import decoupled_fit_with_cross_val_stage_two


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

    def _fresh_callbacks(self, early_stop_cb):
        if early_stop_cb is None:
            return None
        # Keras expects a list[Callback], not a nested list
        return [clone_early_stopping(early_stop_cb)]

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

    def model_prepare(self, model: Any) -> Any:
        representation_layer_index = backend.tools.positive_model_layer_index(model, model._representation_layer_index)
        found_layer, found_index = imbal.util.get_representation_layer_index(
            model,
            desired_layer_index=representation_layer_index
        )
        if found_index is None:
            raise ValueError(
                "Unable to find viable representation layer. Please ensure you model has at least two trainable layers")
        if representation_layer_index > found_index:
            warnings.warn(
                f"Overriding representation layer to layer {found_index} (originally {representation_layer_index})")
            representation_layer_index = found_index

        untrainable_layers = model.layers[:representation_layer_index + 1]
        trainable_layers = model.layers[representation_layer_index + 1:]

        for layer in trainable_layers:
            if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
                layer.set_weights([layer.kernel_initializer(shape=np.asarray(layer.kernel.shape)),
                                   layer.bias_initializer(shape=np.asarray(layer.bias.shape))])
        for layer in untrainable_layers:
            layer.trainable = False

        if model._use_decoder_branch:
            for layer in model._decoder_branch:
                layer.trainable = False

        return model

    def finalize_model_after_training(self, model):
        model._use_decoder_branch = model._generate_decoder_branch

        if model._generate_decoder_branch:
            model._extended_model.trainable = True


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
):
    """
    Modular k-fold driver. Uses:
      - strategy: decides which fit method to call + how to handle special logic
      - params: a union params object specific to Regular/Balanced/Decoupled
    """

    # Pull early stopping from params.callbacks (if present)
    callbacks = params.callbacks or []
    early_stop = callbacks[0] if callbacks else None

    # Driver overrides batch_size if you want it centralized
    params = params.copy()
    params.batch_size = batch_size

    model = strategy.model_prepare(model)

    compile_cfg = model.get_compile_config()
    initial_weights = model.get_weights()

    # Fit callable (lambda)
    fit_call = strategy.make_call(model)

    best_epochs: List[int] = []

    # NOTE: adapt to your stratified_kfold_indices signature
    for fold, (tr_idx, va_idx) in enumerate(
        stratified_kfold_indices(y, k=num_folds, seed=seed, mode=mode, shuffle=shuffle)
    ):

        print(model._use_decoder_branch)
        # This recreates a *fresh optimizer + loss + metrics* based on the saved config
        model.compile_from_config(compile_cfg)
        print(model._use_decoder_branch)

        model.set_weights(initial_weights)

        # optional: clear metric state (compile_from_config usually recreates them anyway)
        model.reset_metrics()

        fold_params, _ = strategy.fold_prepare(
            params,
            tr_idx=tr_idx,
            va_idx=va_idx,
            x=x,
            y=y,
            sample_weight=params.sample_weight,
            early_stop_cb=early_stop,
        )

        fold_fit_kwargs = fold_params.to_fit_kwargs_for_fold(max_epochs=DEFAULT_MAX_EPOCHS)

        history = fit_call(fold_fit_kwargs)

        val_losses = strategy.extract_val_losses(history)
        if val_losses:
            best_epoch = int(np.argmin(val_losses))
            best_epochs.append(best_epoch + 1)

    avg_best_epoch = int(np.rint(np.mean(best_epochs))) if best_epochs else 1
    avg_best_epoch = max(1, avg_best_epoch)
    print("Best epoch (average):", avg_best_epoch)

    model.set_weights(initial_weights)

    # This recreates a *fresh optimizer + loss + metrics* based on the saved config
    model.compile_from_config(compile_cfg)

    # optional: clear metric state (compile_from_config usually recreates them anyway)
    model.reset_metrics()

    full_params = params.copy()
    full_params.x = x
    full_params.y = y
    full_fit_kwargs = full_params.to_fit_kwargs_for_full_train(epochs=avg_best_epoch)

    full_history = fit_call(full_fit_kwargs)

    model = strategy.finalize_model_after_training(model)

    return full_history

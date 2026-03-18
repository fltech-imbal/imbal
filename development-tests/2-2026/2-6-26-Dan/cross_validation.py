import numpy as np
import keras

import imbal
from sklearn.model_selection import StratifiedKFold
from stratified_split_for_k_folds import stratified_kfold_indices
from imbal.util.backend.constants import ModelType

DEFAULT_MAX_EPOCHS = 100_000


def fit_k_folds(model,
                x,
                y,
                batch_size=64,
                num_folds=5,
                shuffle=True,
                seed=0,
                mode=ModelType.CLASSIFICATION,
                **kwargs):
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

    def normalize_sample_weights(sw, eps=1e-8):
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

    fold_kwargs = kwargs.copy()
    fold_kwargs.pop("sample_weight", None)

    # check for stratified sampling
    if y is None:
        x, y, sample_weight = x.unpack()
        print("AFTER UNPACK:", y.shape, "pos", int(y.sum()), "neg", int((1 - y).sum()))

        kwargs.pop("sample_weight", None)
    else:
        sample_weight = kwargs.pop("sample_weight", None)

    best_epochs = []

    original_model_weights = model.get_weights()

    callbacks = fold_kwargs.pop("callbacks", None)
    early_stop = callbacks[0] if callbacks else None

    for fold, (tr_idx, va_idx) in enumerate(stratified_kfold_indices(y, k=num_folds, seed=seed, mode=mode, shuffle=shuffle)):
        model.set_weights(original_model_weights)
        fresh_early_stopping = [clone_early_stopping(early_stop)] if early_stop else None

        X_tr, X_va = x[tr_idx], x[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        sw_tr = normalize_sample_weights(sample_weight[tr_idx])
        sw_va = normalize_sample_weights(sample_weight[va_idx])

        history = model.fit(
            X_tr,
            y_tr,
            sample_weight=sw_tr,
            validation_data=(X_va, y_va, sw_va),
            batch_size=batch_size,
            callbacks=[fresh_early_stopping],
            epochs=DEFAULT_MAX_EPOCHS,
            stratify_batches=True,
            **{k: v for k, v in fold_kwargs.items() if k != "epochs"},
        )

        val_losses = history.history["val_loss"]
        if val_losses:
            best_epoch = int(np.argmin(val_losses))
            best_epochs.append(best_epoch + 1)

    avg_best_epoch = int(np.rint(np.mean(best_epochs)))
    avg_best_epoch = max(1, avg_best_epoch)

    print("Best epoch (average):", avg_best_epoch)

    model.set_weights(original_model_weights)

    full_train_history = model.fit(
        x,
        y,
        batch_size=batch_size,
        sample_weight=sample_weight,
        epochs=avg_best_epoch,
        stratify_batches=True,
        **{k: v for k, v in kwargs.items() if k != "epochs" and k != "callbacks"},
    )

    return full_train_history

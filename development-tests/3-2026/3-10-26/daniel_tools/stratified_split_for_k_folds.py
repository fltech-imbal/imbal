from __future__ import annotations

from typing import Generator, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

from imbal.util.backend.constants import ModelType


def _make_regression_pseudo_classes(
    y: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Convert continuous y into pseudo-classes by sorting and assigning
    consecutive ranks into bins (size 10 or 100), similar to your current logic.

    Returns: array of ints (same length as y), representing bin IDs.
    """
    y = np.asarray(y).reshape(-1)
    n = y.shape[0]

    # Choose batch size (10 vs 100) using a fold-friendly heuristic.
    # Your old choice depended on train_size being multiple of 0.1; for k-folds
    # we can use whether 1/k is multiple of 0.1 as an analogue.
    val_size = 1.0 / k
    # If validation fraction is like 0.1, 0.2, 0.3... use 10, else 100
    if abs(val_size * 10 - round(val_size * 10)) < 1e-6:
        batch_size = 10
    else:
        batch_size = 100

    # Sort by y, then assign bin IDs along the sorted order
    sort_order = np.argsort(y, kind="mergesort")  # stable sort helps with ties
    ranks = np.empty(n, dtype=int)
    ranks[sort_order] = np.arange(n)

    pseudo = ranks // batch_size
    return pseudo


def stratified_kfold_indices(
    y: np.ndarray,
    k: int = 5,
    *,
    seed: Optional[int] = None,
    shuffle: bool = True,
    mode=ModelType.CLASSIFICATION,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yield (train_idx, val_idx) for each fold.

    - classification: stratify on y directly
    - regression: stratify on pseudo-classes derived from sorted y batches
    """
    y = np.asarray(y).reshape(-1)
    n = y.shape[0]
    all_idx = np.arange(n)

    if k < 2:
        raise ValueError("k must be >= 2")
    if k > n:
        raise ValueError("k cannot exceed number of samples")

    if mode == ModelType.CLASSIFICATION:
        splitter = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
        for train_idx, val_idx in splitter.split(all_idx, y):
            yield train_idx, val_idx

    elif mode == ModelType.REGRESSION:
        pseudo = _make_regression_pseudo_classes(y, k=k)

        # If pseudo-classes are too fine, some bins may have < k members,
        # which breaks StratifiedKFold. We can coarsen bins automatically.
        # (This is the most common failure mode for stratified regression.)
        while True:
            # Count examples per pseudo-class
            _, counts = np.unique(pseudo, return_counts=True)
            if counts.min() >= k:
                break
            # Coarsen by merging adjacent bins: divide bin id by 2
            pseudo = pseudo // 2
            # If everything collapses to one bin, fall back to plain KFold
            if np.unique(pseudo).size == 1:
                splitter = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
                for train_idx, val_idx in splitter.split(all_idx):
                    yield train_idx, val_idx
                return

        splitter = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
        for train_idx, val_idx in splitter.split(all_idx, pseudo):
            yield train_idx, val_idx

    else:
        raise ValueError(f"Unknown mode: {mode}")

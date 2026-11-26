"""Expanding time-window cross-validation and minority-preserving subsampling."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _factorize_sorted_periods(time_values: pd.Series | pd.Index | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map sortable time values to dense 0..P-1 period indices in ascending chronological order."""
    arr = pd.Series(time_values).reset_index(drop=True)
    if arr.isna().any():
        raise ValueError("time_values must not contain missing (NaN/NaT/None) values.")
    codes, uniques = pd.factorize(arr, sort=True)
    return codes.astype(np.int64, copy=False), np.asarray(uniques)


def make_expanding_time_splits(
    time_values: pd.Series | pd.Index | np.ndarray,
    n_splits: int = 5,
    min_train_periods: int | None = None,
    val_periods: int | None = None,
    gap_periods: int = 0,
    oot_periods: int = 0,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray | None]:
    """Create expanding-window time-series CV splits and an optional untouched OOT holdout.

    Splits strictly along unique sorted values of ``time_values`` so rows sharing the
    same time period never appear in both train and validation.

    Parameters
    ----------
    time_values:
        1D sequence of sortable timestamps, dates, or vintage identifiers (e.g., YYYYMM).
    n_splits:
        Number of expanding CV folds (default: 5).
    min_train_periods:
        Number of initial unique periods in fold 0's training set. Computed automatically
        when None.
    val_periods:
        Number of unique periods in each fold's validation window. Computed automatically
        when None.
    gap_periods:
        Number of embargo/gap periods skipped between the end of train and start of val.
    oot_periods:
        Number of final unique periods reserved as an untouched out-of-time holdout set.

    Returns
    -------
    cv_splits:
        List of ``(train_idx, val_idx)`` tuples of integer positional indices.
    oot_idx:
        Integer positional indices for the OOT holdout, or ``None`` if ``oot_periods == 0``.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}.")
    if gap_periods < 0:
        raise ValueError(f"gap_periods must be >= 0, got {gap_periods}.")
    if oot_periods < 0:
        raise ValueError(f"oot_periods must be >= 0, got {oot_periods}.")

    codes, uniques = _factorize_sorted_periods(time_values)
    total_periods = len(uniques)
    dev_periods = total_periods - oot_periods

    if dev_periods < n_splits + 1 + gap_periods:
        raise ValueError(
            f"Not enough unique time periods ({total_periods}) for n_splits={n_splits}, "
            f"gap_periods={gap_periods}, oot_periods={oot_periods}."
        )

    avail = dev_periods - gap_periods
    if val_periods is None and min_train_periods is None:
        val_periods = max(1, avail // (n_splits + 2)) if avail >= 2 * n_splits + 1 else max(1, avail // (n_splits + 1))
        min_train_periods = avail - n_splits * val_periods
    elif val_periods is None:
        assert min_train_periods is not None
        val_periods = (avail - min_train_periods) // n_splits
    elif min_train_periods is None:
        min_train_periods = avail - n_splits * val_periods

    if min_train_periods < 1 or val_periods < 1:
        raise ValueError(
            f"Invalid period allocation: min_train_periods={min_train_periods}, "
            f"val_periods={val_periods} (dev_periods={dev_periods}, gap_periods={gap_periods})."
        )
    if min_train_periods + gap_periods + n_splits * val_periods > dev_periods:
        raise ValueError(
            f"Requested expanding schedule requires {min_train_periods + gap_periods + n_splits * val_periods} "
            f"development periods, but only {dev_periods} are available."
        )

    cv_splits: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        train_end_period = min_train_periods + k * val_periods
        val_start_period = train_end_period + gap_periods
        val_end_period = val_start_period + val_periods

        train_mask = codes < train_end_period
        val_mask = (codes >= val_start_period) & (codes < val_end_period)

        train_idx = np.flatnonzero(train_mask).astype(np.int64, copy=False)
        val_idx = np.flatnonzero(val_mask).astype(np.int64, copy=False)
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"Fold {k} produced an empty split (train={len(train_idx)}, val={len(val_idx)}).")
        cv_splits.append((train_idx, val_idx))

    oot_idx: np.ndarray | None = None
    if oot_periods > 0:
        oot_mask = codes >= dev_periods
        oot_idx = np.flatnonzero(oot_mask).astype(np.int64, copy=False)

    return cv_splits, oot_idx


def validate_cv_splits(
    cv_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    n_samples: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validate and normalize ``cv_splits`` into 1D int64 index arrays."""
    if len(cv_splits) == 0:
        raise ValueError("cv_splits must contain at least one fold.")

    normalized: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_id, pair in enumerate(cv_splits):
        if len(pair) != 2:
            raise ValueError(f"Fold {fold_id} must be a (train_idx, val_idx) tuple.")
        tr = np.asarray(pair[0], dtype=np.int64).ravel()
        va = np.asarray(pair[1], dtype=np.int64).ravel()
        if len(tr) == 0 or len(va) == 0:
            raise ValueError(f"Fold {fold_id} has empty train ({len(tr)}) or val ({len(va)}) indices.")
        if tr.min() < 0 or tr.max() >= n_samples or va.min() < 0 or va.max() >= n_samples:
            raise ValueError(f"Fold {fold_id} contains indices out of bounds [0, {n_samples}).")
        if np.intersect1d(tr, va, assume_unique=False).size > 0:
            raise ValueError(f"Fold {fold_id} has overlapping indices between train and val.")
        normalized.append((tr, va))
    return normalized


def _sample_negatives_by_time(
    neg_idx: np.ndarray,
    neg_periods: np.ndarray | None,
    n_target: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n_target`` negatives proportionally across time periods and compute inverse-propensity weights."""
    n_neg = len(neg_idx)
    if n_target >= n_neg:
        return neg_idx, np.ones(n_neg, dtype=np.float32)

    if neg_periods is None:
        chosen_pos = rng.choice(n_neg, size=n_target, replace=False)
        chosen_pos.sort()
        weight = np.float32(n_neg / len(chosen_pos))
        return neg_idx[chosen_pos], np.full(len(chosen_pos), weight, dtype=np.float32)

    unique_p, inverse, counts = np.unique(neg_periods, return_inverse=True, return_counts=True)
    # Allocate integer sample sizes per period using largest-remainder rounding
    exact = counts * (n_target / n_neg)
    alloc = np.floor(exact).astype(np.int64)
    # Ensure periods with negatives get at least 1 row whenever budget permits
    nonzero_mask = counts > 0
    if np.sum(alloc == 0) > 0 and n_target >= int(np.sum(nonzero_mask)):
        zero_needed = nonzero_mask & (alloc == 0)
        alloc[zero_needed] = 1

    diff = n_target - int(alloc.sum())
    if diff > 0:
        remainders = exact - alloc
        remainders[alloc >= counts] = -1.0
        order = np.argsort(-remainders)
        for p_i in order:
            if diff == 0:
                break
            if alloc[p_i] < counts[p_i]:
                alloc[p_i] += 1
                diff -= 1
    elif diff < 0:
        # Trim excess from largest allocations above 1
        order = np.argsort(-alloc)
        for p_i in order:
            if diff == 0:
                break
            reducible = max(0, int(alloc[p_i]) - 1)
            take = min(reducible, -diff)
            alloc[p_i] -= take
            diff += take

    chosen_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for p_i in range(len(unique_p)):
        k = int(alloc[p_i])
        if k <= 0:
            continue
        loc = np.flatnonzero(inverse == p_i)
        if k >= len(loc):
            picked = loc
        else:
            picked = rng.choice(loc, size=k, replace=False)
        chosen_parts.append(neg_idx[picked])
        weight_parts.append(np.full(len(picked), counts[p_i] / len(picked), dtype=np.float32))

    sampled_idx = np.concatenate(chosen_parts)
    sampled_w = np.concatenate(weight_parts)
    order = np.argsort(sampled_idx)
    return sampled_idx[order], sampled_w[order]


def stratified_time_subsample(
    train_idx: np.ndarray,
    y: np.ndarray,
    time_values: np.ndarray | None = None,
    max_rows: int = 300_000,
    keep_all_positives: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Downsample training indices while preserving positive cases and temporal distribution.

    When ``keep_all_positives=True`` and positives make up fewer than half of ``max_rows``,
    100% of positive rows are kept (weight 1.0) and negative rows are sampled proportionally
    across time periods with inverse-sampling weights so the effective class prior matches
    the full training fold.
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if max_rows <= 0 or len(train_idx) <= max_rows:
        return np.sort(train_idx), None

    rng = np.random.default_rng(random_state)
    y_tr = np.asarray(y)[train_idx]
    pos_mask = y_tr == 1
    n_pos = int(np.sum(pos_mask))
    n_neg = len(train_idx) - n_pos

    if keep_all_positives and 0 < n_pos < int(max_rows * 0.5) and n_neg > 0:
        pos_idx = train_idx[pos_mask]
        neg_idx = train_idx[~pos_mask]
        neg_periods = np.asarray(time_values)[neg_idx] if time_values is not None else None
        n_neg_target = max(1, max_rows - n_pos)
        sampled_neg_idx, sampled_neg_w = _sample_negatives_by_time(
            neg_idx, neg_periods, n_neg_target, rng
        )
        combined_idx = np.concatenate([pos_idx, sampled_neg_idx])
        combined_w = np.concatenate(
            [np.ones(len(pos_idx), dtype=np.float32), sampled_neg_w]
        )
        order = np.argsort(combined_idx)
        return combined_idx[order], combined_w[order]

    # Balanced fallback: proportional stratified sample across (time, y) or y
    if time_values is not None:
        periods = np.asarray(time_values)[train_idx]
        strata = pd.MultiIndex.from_arrays([periods, y_tr]).codes
    else:
        strata = y_tr.astype(np.int64, copy=False)

    sampled_idx, _ = _sample_negatives_by_time(train_idx, strata, max_rows, rng)
    return sampled_idx, None

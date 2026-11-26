"""Memory-safe float32 Fortran-contiguous matrix preparation for LightGBM RFE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class PreparedMatrix:
    """Column-major float32 feature matrix and metadata for zero-copy pandas-free CV."""

    X: np.ndarray
    feature_names: list[str]
    categorical_indices: frozenset[int]
    n_samples: int
    n_features: int


def _encode_column(series: pd.Series, force_categorical: bool) -> tuple[np.ndarray, bool]:
    """Convert a pandas Series to float32 without mutating the caller's DataFrame."""
    dtype = series.dtype
    is_cat = (
        force_categorical
        or isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    )
    if is_cat:
        codes = pd.Categorical(series).codes.astype(np.float32, copy=False)
        if np.any(codes < 0):
            codes = codes.copy()
            codes[codes < 0] = np.nan
        return codes, True

    if pd.api.types.is_bool_dtype(dtype):
        return series.to_numpy(dtype=np.float32, na_value=np.nan), False

    return series.to_numpy(dtype=np.float32, na_value=np.nan), False


def prepare_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    categorical_cols: Sequence[str] | None = None,
) -> PreparedMatrix:
    """Convert a pandas DataFrame into a float32 Fortran-contiguous array column-by-column.

    Allocating column-major ``float32`` directly avoids creating an intermediate ~36 GB
    ``float64`` copy on 1.3M x 3,500 datasets and makes column subsetting ``X[:, cols]``
    fast across RFE rounds.
    """
    cols = list(df.columns) if feature_cols is None else list(feature_cols)
    if not cols:
        raise ValueError("feature_cols must contain at least one feature.")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing[:10]}")

    cat_set = set(categorical_cols or ())
    n_samples = len(df)
    n_features = len(cols)
    X = np.empty((n_samples, n_features), dtype=np.float32, order="F")
    cat_indices: set[int] = set()

    for j, col in enumerate(cols):
        arr, is_cat = _encode_column(df[col], force_categorical=(col in cat_set))
        X[:, j] = arr
        if is_cat:
            cat_indices.add(j)

    return PreparedMatrix(
        X=X,
        feature_names=cols,
        categorical_indices=frozenset(cat_indices),
        n_samples=n_samples,
        n_features=n_features,
    )


def extract_step_matrix(
    prepared: PreparedMatrix,
    active_indices: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Materialize a C-contiguous float32 slice for the current active feature subset.

    Called once per RFE round and shared across all expanding folds for that round.
    """
    idx = np.asarray(active_indices, dtype=np.int64)
    X_step = np.ascontiguousarray(prepared.X[:, idx], dtype=np.float32)
    local_cat_indices = [
        local_j for local_j, global_j in enumerate(idx.tolist()) if global_j in prepared.categorical_indices
    ]
    return X_step, local_cat_indices

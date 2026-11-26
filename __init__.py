"""Staged LightGBM Recursive Feature Elimination for large tabular datasets."""

from .data import PreparedMatrix, extract_step_matrix, prepare_matrix
from .plotting import plot_fold_stability, plot_rfe_curve, plot_top_features
from .selector import StagedLGBMRFE, StagedRFEConfig
from .splits import make_expanding_time_splits, stratified_time_subsample, validate_cv_splits

__all__ = [
    "PreparedMatrix",
    "StagedLGBMRFE",
    "StagedRFEConfig",
    "extract_step_matrix",
    "make_expanding_time_splits",
    "plot_fold_stability",
    "plot_rfe_curve",
    "plot_top_features",
    "prepare_matrix",
    "stratified_time_subsample",
    "validate_cv_splits",
]

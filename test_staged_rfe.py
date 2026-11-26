"""Comprehensive test suite for lgbm_staged_rfe."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lgbm_staged_rfe import (
    StagedLGBMRFE,
    StagedRFEConfig,
    make_expanding_time_splits,
    plot_fold_stability,
    plot_rfe_curve,
    plot_top_features,
    prepare_matrix,
    stratified_time_subsample,
    validate_cv_splits,
)


def _make_synthetic_df(
    n_rows: int = 4_000,
    n_noise: int = 65,
    n_periods: int = 12,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Build an ~8% positive synthetic temporal dataset with signal, collinearity, noise, and categorical."""
    rng = np.random.default_rng(seed)
    periods = np.repeat(np.arange(1, n_periods + 1), n_rows // n_periods)
    if len(periods) < n_rows:
        periods = np.concatenate([periods, np.full(n_rows - len(periods), n_periods)])

    n_signal = 8
    X_sig = rng.normal(0.0, 1.0, size=(n_rows, n_signal)).astype(np.float32)
    # Add 4 near-collinear duplicates of sig_0..sig_3
    X_dup = X_sig[:, :4] + rng.normal(0.0, 0.05, size=(n_rows, 4)).astype(np.float32)
    X_noise = rng.normal(0.0, 1.0, size=(n_rows, n_noise)).astype(np.float32)

    latent = (
        1.4 * X_sig[:, 0]
        - 1.2 * X_sig[:, 1]
        + 1.0 * np.tanh(X_sig[:, 2] * 1.4)
        + 0.9 * (X_sig[:, 3] * X_sig[:, 4])
        - 0.8 * X_sig[:, 5]
        + 0.7 * X_sig[:, 6]
        + 0.6 * X_sig[:, 7]
        + rng.normal(0.0, 0.4, size=n_rows)
    )
    probs = 1.0 / (1.0 + np.exp(-(latent - 2.95)))
    y = rng.binomial(1, probs).astype(np.int8)

    data: dict[str, object] = {}
    for i in range(n_signal):
        data[f"sig_{i}"] = X_sig[:, i]
    for i in range(4):
        data[f"dup_{i}"] = X_dup[:, i]
    for i in range(n_noise):
        data[f"noise_{i}"] = X_noise[:, i]

    cat_vals = np.where(latent + rng.normal(0, 1.5, n_rows) > 0.3, "high", "low").astype(object)
    cat_vals[::50] = None
    data["cat_seg"] = pd.Series(cat_vals, dtype="category")

    df = pd.DataFrame(data)
    true_signals = [f"sig_{i}" for i in range(n_signal)]
    return df, y, periods, true_signals


def test_make_expanding_time_splits_monotonic_and_oot() -> None:
    periods = np.repeat(np.arange(202401, 202415), 100)  # 14 unique periods
    cv_splits, oot_idx = make_expanding_time_splits(
        periods,
        n_splits=5,
        gap_periods=1,
        oot_periods=2,
    )
    assert len(cv_splits) == 5
    assert oot_idx is not None
    assert len(oot_idx) == 200
    assert set(periods[oot_idx]) == {202413, 202414}

    prev_train_len = 0
    for tr_idx, va_idx in cv_splits:
        assert len(tr_idx) > prev_train_len
        prev_train_len = len(tr_idx)
        assert periods[tr_idx].max() + 1 < periods[va_idx].min()  # gap_periods=1 respected
        assert np.intersect1d(tr_idx, va_idx).size == 0
        assert np.intersect1d(tr_idx, oot_idx).size == 0
        assert np.intersect1d(va_idx, oot_idx).size == 0


def test_make_expanding_time_splits_validation_errors() -> None:
    with pytest.raises(ValueError, match="missing"):
        make_expanding_time_splits(np.array([1.0, np.nan, 2.0]), n_splits=2)

    with pytest.raises(ValueError, match="Not enough"):
        make_expanding_time_splits(np.arange(5), n_splits=5)

    with pytest.raises(ValueError, match="overlapping"):
        validate_cv_splits([(np.array([0, 1, 2]), np.array([2, 3]))], n_samples=5)


def test_stratified_time_subsample_preserves_positives_and_weights() -> None:
    rng = np.random.default_rng(7)
    n = 20_000
    periods = np.repeat(np.arange(10), n // 10)
    y = rng.binomial(1, 0.08, size=n)
    train_idx = np.arange(n)

    sub_idx, weights = stratified_time_subsample(
        train_idx,
        y,
        time_values=periods,
        max_rows=5_000,
        keep_all_positives=True,
        random_state=42,
    )
    assert len(sub_idx) == 5_000
    assert weights is not None
    assert np.sum(y[sub_idx] == 1) == np.sum(y == 1)
    assert np.allclose(weights[y[sub_idx] == 1], 1.0)
    # Effective weighted negative count matches full training negative count
    assert np.isclose(weights[y[sub_idx] == 0].sum(), np.sum(y == 0), rtol=1e-4)

    # Passthrough when already below max_rows
    full_idx, full_w = stratified_time_subsample(train_idx[:1000], y, max_rows=5_000)
    assert len(full_idx) == 1000
    assert full_w is None


def test_prepare_matrix_float32_f_order_and_non_mutating() -> None:
    df = pd.DataFrame(
        {
            "num_f64": [1.5, 2.5, np.nan, 4.0],
            "flag": [True, False, True, False],
            "cat_str": ["a", "b", None, "a"],
        }
    )
    df_orig = df.copy(deep=True)
    prep = prepare_matrix(df)

    pd.testing.assert_frame_equal(df, df_orig)
    assert prep.X.dtype == np.float32
    assert prep.X.flags["F_CONTIGUOUS"]
    assert prep.categorical_indices == frozenset({2})
    assert np.isnan(prep.X[2, 0])
    assert np.isnan(prep.X[2, 2])
    assert prep.X[0, 1] == 1.0


def test_staged_rfe_geometric_shap_end_to_end(tmp_path: Path) -> None:
    df, y, periods, true_signals = _make_synthetic_df(n_rows=4_000, n_noise=65, n_periods=12)
    cv_splits, oot_idx = make_expanding_time_splits(periods, n_splits=5, oot_periods=2)
    assert oot_idx is not None

    dev_mask = np.ones(len(df), dtype=bool)
    dev_mask[oot_idx] = False
    dev_df = df.iloc[dev_mask].reset_index(drop=True)
    dev_y = y[dev_mask]
    dev_periods = periods[dev_mask]
    cv_dev, _ = make_expanding_time_splits(dev_periods, n_splits=5)

    oot_df = df.iloc[oot_idx].reset_index(drop=True)
    oot_y = y[oot_idx]

    cfg = StagedRFEConfig(
        stage1_method="geometric_shap_rfe",
        stage1_target=35,
        stage1_drop_fraction=0.30,
        stage1_max_train_rows=1_200,  # forces subsampling + full-data baseline
        stage2_target=16,
        stage2_drop_fraction=0.25,
        stage3_step=2,
        min_features=8,
        ranking_mode="val_shap_stability",
        num_boost_round=120,
        early_stopping_rounds=20,
        n_jobs=2,
        checkpoint_dir=tmp_path / "ckpt",
        verbose=False,
    )
    selector = StagedLGBMRFE(cfg).fit(dev_df, dev_y, cv_splits=cv_dev, time_values=dev_periods)

    assert not selector.results_.empty
    stages = set(selector.results_["stage"])
    assert {"baseline_full", "stage1_coarse", "stage2_medium", "stage3_fine"}.issubset(stages)
    assert selector.baseline_result_ is not None

    # Signal features should dominate top 16
    feats_16 = selector.get_features(16)
    assert len(feats_16) == 16
    recovered = len(set(feats_16) & set(true_signals))
    assert recovered >= 6

    rec_1se = selector.suggest_n_features(method="one_se")
    rec_delta = selector.suggest_n_features(method="delta_auc", min_delta_auc=0.002)
    rec_max = selector.suggest_n_features(method="max")
    assert rec_1se["n_features"] <= rec_max["n_features"]
    assert rec_delta["n_features"] <= rec_max["n_features"]

    oot_df_res = selector.evaluate_oot(dev_df, dev_y, oot_df, oot_y, n_features_list=[16, 8])
    assert len(oot_df_res) == 2
    assert (oot_df_res["oot_auc"] > 0.70).all()

    # Check plotting functions render & save clean PNGs
    p1 = tmp_path / "curve.png"
    p2 = tmp_path / "stability.png"
    p3 = tmp_path / "top_feats.png"
    plot_rfe_curve(
        selector.results_,
        baseline_auc=selector.baseline_result_["mean_auc"],
        target_budget=16,
        zoom_max_features=35,
        save_path=p1,
    )
    plot_fold_stability(selector.results_, zoom_max_features=35, target_budget=16, save_path=p2)
    plot_top_features(selector.get_feature_importances(16), top_n=16, save_path=p3)

    for p in (p1, p2, p3):
        assert p.exists() and p.stat().st_size > 5_000


def test_null_importance_and_collinearity_pruning() -> None:
    df, y, periods, true_signals = _make_synthetic_df(n_rows=3_000, n_noise=45, n_periods=10, seed=99)
    cv_splits, _ = make_expanding_time_splits(periods, n_splits=3)

    cfg = StagedRFEConfig(
        stage1_method="null_importance",
        null_importance_runs=5,
        stage1_target=25,
        stage1_max_train_rows=10_000,
        collinearity_threshold=0.85,
        stage2_target=12,
        stage2_drop_fraction=0.30,
        stage3_step=2,
        min_features=8,
        num_boost_round=80,
        early_stopping_rounds=15,
        n_jobs=2,
        verbose=False,
    )
    selector = StagedLGBMRFE(cfg).fit(df, y, cv_splits=cv_splits, time_values=periods)
    assert "stage1_null_importance" in set(selector.results_["stage"])
    feats_12 = selector.get_features(12)
    assert len(feats_12) == 12
    # Collinear pairs (sig_i, dup_i) should be deduplicated into at most one representative
    for i in range(4):
        assert not (f"sig_{i}" in feats_12 and f"dup_{i}" in feats_12)
    # True signal dimensions (via sig_i or its near-identical representative dup_i) should dominate
    recovered_dims = sum(
        1 for i in range(len(true_signals)) if (f"sig_{i}" in feats_12 or f"dup_{i}" in feats_12)
    )
    assert recovered_dims >= 6


def test_checkpoint_resume_consistency(tmp_path: Path) -> None:
    df, y, periods, _ = _make_synthetic_df(n_rows=2_400, n_noise=30, n_periods=8, seed=11)
    cv_splits, _ = make_expanding_time_splits(periods, n_splits=3)
    ckpt_dir = tmp_path / "resume_ckpt"

    cfg = StagedRFEConfig(
        stage1_target=20,
        stage1_drop_fraction=0.30,
        stage1_max_train_rows=5_000,
        stage2_target=12,
        stage2_drop_fraction=0.25,
        stage3_step=2,
        min_features=8,
        num_boost_round=60,
        early_stopping_rounds=15,
        n_jobs=2,
        checkpoint_dir=ckpt_dir,
        verbose=False,
    )
    full_selector = StagedLGBMRFE(cfg).fit(df, y, cv_splits=cv_splits, time_values=periods)

    # Simulate partial checkpoint truncated at Stage 2 by slicing saved files
    hist = pd.read_csv(ckpt_dir / "history.csv")
    partial_hist = hist[hist["n_features"] >= 14].copy()
    partial_hist.to_csv(ckpt_dir / "history.csv", index=False)

    fsets = json.loads((ckpt_dir / "feature_sets.json").read_text(encoding="utf-8"))
    partial_fsets = {k: v for k, v in fsets.items() if int(k) >= 14}
    (ckpt_dir / "feature_sets.json").write_text(json.dumps(partial_fsets), encoding="utf-8")

    # Next active subset is at n=12 (derived from n=14 step)
    active_12 = full_selector.get_features(12)
    state = json.loads((ckpt_dir / "state.json").read_text(encoding="utf-8"))
    state["stage"] = "stage2_medium"
    state["active_features"] = active_12
    (ckpt_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")

    resumed_selector = StagedLGBMRFE(cfg).fit(df, y, cv_splits=cv_splits, time_values=periods, resume=True)
    assert resumed_selector.get_features(8) == full_selector.get_features(8)
    assert list(resumed_selector.results_["n_features"]) == list(full_selector.results_["n_features"])

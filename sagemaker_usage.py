"""Example end-to-end workflow for StagedLGBMRFE (synthetic demo + SageMaker template).

Run locally in seconds:
    python examples/sagemaker_usage.py --synthetic --output-dir output_demo
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

# Allow running directly from the repo root without pip install -e .
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
)


def make_synthetic_imbalanced_dataset(
    n_rows: int = 8_000,
    n_features: int = 120,
    n_signal: int = 10,
    n_periods: int = 14,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Generate a synthetic ~8% positive binary classification dataset across time vintages."""
    rng = np.random.default_rng(random_state)
    periods = np.repeat(np.arange(202401, 202401 + n_periods), n_rows // n_periods)
    if len(periods) < n_rows:
        periods = np.concatenate([periods, np.full(n_rows - len(periods), 202401 + n_periods - 1)])

    X = rng.normal(0.0, 1.0, size=(n_rows, n_features)).astype(np.float32)

    # Strong non-linear + linear signal in the first n_signal columns
    latent = (
        1.25 * X[:, 0]
        - 1.10 * X[:, 1]
        + 0.90 * np.tanh(X[:, 2] * 1.5)
        + 0.85 * (X[:, 3] * X[:, 4])
        + 0.70 * np.maximum(X[:, 5], 0.0)
        - 0.65 * X[:, 6]
        + 0.60 * X[:, 7]
        - 0.55 * X[:, 8]
        + 0.50 * X[:, 9]
        + rng.normal(0.0, 0.5, size=n_rows)
    )
    # Calibrate intercept for ~8% positive base rate
    logits = latent - 2.95
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs).astype(np.int8)

    cols = [f"sig_{i}" if i < n_signal else f"noise_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["segment_cat"] = pd.Categorical(
        np.where(latent + rng.normal(0, 1.2, n_rows) > 0.4, "tier_a", "tier_b")
    )
    feature_cols = cols + ["segment_cat"]
    return df, y, periods, feature_cols


def run_synthetic_demo(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df, y, time_values, feature_cols = make_synthetic_imbalanced_dataset()
    print(f"Dataset: {df.shape[0]:,} rows x {len(feature_cols)} features | positive rate={y.mean():.2%}")

    # Reserve last 2 periods as untouched OOT holdout; 5 expanding CV folds on earlier periods
    cv_splits, oot_idx = make_expanding_time_splits(
        time_values,
        n_splits=5,
        oot_periods=2,
    )
    assert oot_idx is not None
    dev_mask = np.ones(len(df), dtype=bool)
    dev_mask[oot_idx] = False

    # Build CV splits indexed relative to dev_df (or pass full df and cv_splits directly)
    dev_df = df.iloc[dev_mask].reset_index(drop=True)
    dev_y = y[dev_mask]
    dev_time = time_values[dev_mask]
    cv_splits_dev, _ = make_expanding_time_splits(dev_time, n_splits=5, oot_periods=0)

    oot_df = df.iloc[oot_idx].reset_index(drop=True)
    oot_y = y[oot_idx]

    # Pre-convert to float32 Fortran-contiguous matrix once
    prepared_dev = prepare_matrix(dev_df, feature_cols=feature_cols)

    config = StagedRFEConfig(
        stage1_method="geometric_shap_rfe",
        stage1_target=60,
        stage1_drop_fraction=0.25,
        stage1_max_train_rows=3_000,
        stage2_target=25,
        stage2_drop_fraction=0.20,
        stage3_step=1,
        min_features=6,
        ranking_mode="val_shap_stability",
        stability_penalty=0.25,
        collinearity_threshold=0.85,
        num_boost_round=250,
        early_stopping_rounds=25,
        n_jobs=4,
        checkpoint_dir=output_dir / "checkpoints",
        verbose=True,
    )

    selector = StagedLGBMRFE(config).fit(
        prepared_dev,
        dev_y,
        cv_splits=cv_splits_dev,
        time_values=dev_time,
    )

    one_se = selector.suggest_n_features(method="one_se")
    delta_rec = selector.suggest_n_features(method="delta_auc", min_delta_auc=0.001)
    peak_rec = selector.suggest_n_features(method="max")
    print("\n--- Selection Summary ---")
    print(f"Peak CV AUC : n={peak_rec['n_features']} (AUC={peak_rec['mean_auc']:.5f})")
    print(f"1-SE Rule   : n={one_se['n_features']} (AUC={one_se['mean_auc']:.5f})")
    print(f"Delta<=0.001: n={delta_rec['n_features']} (AUC={delta_rec['mean_auc']:.5f})")

    baseline_auc = selector.baseline_result_["mean_auc"] if selector.baseline_result_ else None
    plot_rfe_curve(
        selector.results_,
        baseline_auc=baseline_auc,
        target_budget=15,
        zoom_max_features=45,
        save_path=output_dir / "rfe_curve.png",
    )
    plot_fold_stability(
        selector.results_,
        zoom_max_features=45,
        target_budget=15,
        save_path=output_dir / "fold_stability.png",
    )
    plot_top_features(
        selector.get_feature_importances(one_se["n_features"]),
        top_n=one_se["n_features"],
        save_path=output_dir / "top_features.png",
    )

    oot_eval = selector.evaluate_oot(
        dev_df,
        dev_y,
        oot_df,
        oot_y,
        n_features_list=[one_se["n_features"], 15, len(feature_cols)],
    )
    print("\n--- Untouched OOT Evaluation ---")
    print(oot_eval.to_string(index=False))


def sagemaker_production_template() -> None:
    """Reference configuration for 1.3M rows x ~3,500 cols on SageMaker (48 cores, 256-512 GB RAM)."""
    # Example template (not executed directly without your Parquet file):
    #
    # df = pd.read_parquet("s3://your-bucket/features.parquet")
    # feature_cols = [c for c in df.columns if c not in {"target", "vintage_month", "entity_id"}]
    #
    # # 1. Hold out the newest 1-2 months as untouched OOT, build 5 expanding folds on dev
    # cv_splits_all, oot_idx = make_expanding_time_splits(df["vintage_month"], n_splits=5, oot_periods=2)
    # dev_mask = np.ones(len(df), dtype=bool)
    # dev_mask[oot_idx] = False
    # dev_df = df.loc[dev_mask].reset_index(drop=True)
    # oot_df = df.loc[oot_idx].reset_index(drop=True)
    # del df
    #
    # cv_splits, _ = make_expanding_time_splits(dev_df["vintage_month"], n_splits=5)
    # prepared_dev = prepare_matrix(dev_df, feature_cols=feature_cols)  # ~18.2 GB float32 F-order
    #
    # config = StagedRFEConfig(
    #     stage1_method="geometric_shap_rfe",   # or "null_importance"
    #     stage1_target=300,
    #     stage1_drop_fraction=0.25,
    #     stage1_max_train_rows=300_000,        # keeps 100% of ~8% positives + time-stratified negatives
    #     stage1_max_bin=127,
    #     stage1_colsample_bytree=0.25,
    #     stage2_target=70,
    #     stage2_drop_fraction=0.15,
    #     stage2_colsample_bytree=0.50,
    #     stage3_step=1,                        # 1-by-1 precision from 70 -> 15
    #     stage3_colsample_bytree=0.80,
    #     min_features=15,
    #     ranking_mode="val_shap_stability",    # mean(|val_SHAP|) - 0.25 * std(|val_SHAP|)
    #     collinearity_threshold=0.85,          # optional cluster deduplication after Stage 1
    #     n_jobs=48,                            # parallelize inside LightGBM, NOT across CV folds
    #     checkpoint_dir="./rfe_checkpoints",
    # )
    # selector = StagedLGBMRFE(config).fit(
    #     prepared_dev,
    #     dev_df["target"].to_numpy(),
    #     cv_splits=cv_splits,
    #     time_values=dev_df["vintage_month"].to_numpy(),
    # )
    #
    # # Inspect visual trade-off and pick final ~50 features
    # plot_rfe_curve(
    #     selector.results_,
    #     baseline_auc=selector.baseline_result_["mean_auc"],
    #     target_budget=50,
    #     zoom_max_features=120,
    #     save_path="rfe_curve.png",
    # )
    # plot_fold_stability(selector.results_, zoom_max_features=120, target_budget=50, save_path="fold_stability.png")
    # chosen_50 = selector.get_features(50)
    # plot_top_features(selector.get_feature_importances(50), top_n=50, save_path="top_50_features.png")
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StagedLGBMRFE demo")
    parser.add_argument("--synthetic", action="store_true", help="Run fast synthetic demo")
    parser.add_argument("--output-dir", type=Path, default=Path("output_demo"), help="Directory for charts/checkpoints")
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic_demo(args.output_dir)
    else:
        print("Pass --synthetic to execute the local demonstration, or see sagemaker_production_template().")

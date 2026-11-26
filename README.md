# `lgbm_staged_rfe` — Staged LightGBM RFE for 1.3M × 3,500 Datasets

A memory-safe, three-stage coarse-to-fine Recursive Feature Elimination (RFE) pipeline built for **LightGBM** on **large imbalanced binary classification datasets** (~1.3M rows × 3,500 columns, ~8% positives, 5 expanding time folds) running on high-RAM multi-core CPU instances (e.g., SageMaker 48-core / 256–512 GB RAM).

---

## Why Standard `sklearn.RFECV` Fails at This Scale

1. **Compute:** `RFECV(step=1)` with 5 folds requires ~17,500 full model fits.
2. **Memory:** 1.3M × 3,500 in `float64` is ~36.4 GB raw; `RFECV(..., n_jobs=-1)` forks worker processes and pandas copies that quickly blow past RAM or thrash cache.
3. **Methodological Bias:** Standard `RFECV` ranks by *in-sample training* `feature_importances_`, which favors high-cardinality or temporally unstable features.

---

## How `lgbm_staged_rfe` Solves It

| Stage | Feature Range | Rows Used | Step Size | Ranking Signal |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (optional)** | 3,500 | 100% (1.3M) | 1 evaluation | Apples-to-apples CV AUC reference line |
| **Stage 1: Coarse** | 3,500 → ~300 | 300K/fold (100% positives + time-stratified weighted negatives) | 25% geometric drop (or Null Importance) | Validation TreeSHAP stability (`max_bin=127`, `colsample=0.25`) |
| **Optional Collinearity Filter** | ~300 → ~200–250 | 50K sample | Hierarchical Spearman cluster dedup | Keeps top importance feature per tight cluster |
| **Stage 2: Medium** | ~300 → ~70 | 100% (1.3M) | 15% geometric drop | Validation TreeSHAP stability (`mean - 0.25 * std`) |
| **Stage 3: Fine** | ~70 → ~10 | 100% (1.3M) | **1-feature exact step** | Validation TreeSHAP stability across all 5 expanding folds |

### Memory & Speed Optimizations
- **`prepare_matrix()`** pre-allocates a single `float32` Fortran-contiguous NumPy array column-by-column (~18.2 GB for 1.3M × 3,500, avoiding any 36 GB `float64` copy).
- **Sequential folds + `n_jobs=48` inside LightGBM:** Zero duplicate matrix copies across CV worker processes.
- **Validation TreeSHAP (`pred_contrib=True`)** on up to `shap_max_val_rows=50_000` stratified validation rows per fold, while **ROC AUC is always scored on 100% of validation rows**.
- **Checkpointing & Resume:** Saves `history.csv`, `feature_sets.json`, and `state.json` after every step so long SageMaker runs can resume safely.

---

## Quickstart

```python
from lgbm_staged_rfe import (
    StagedLGBMRFE,
    StagedRFEConfig,
    make_expanding_time_splits,
    plot_fold_stability,
    plot_rfe_curve,
    plot_top_features,
    prepare_matrix,
)

# 1. Split off OOT holdout and build 5 expanding time folds
cv_splits, _ = make_expanding_time_splits(dev_df["vintage_month"], n_splits=5)
prepared_dev = prepare_matrix(dev_df, feature_cols=feature_cols)

# 2. Configure and run
config = StagedRFEConfig(
    stage1_target=300,
    stage2_target=70,
    min_features=15,
    n_jobs=48,
    checkpoint_dir="./rfe_checkpoints",
)
selector = StagedLGBMRFE(config).fit(
    prepared_dev,
    dev_df["target"].to_numpy(),
    cv_splits=cv_splits,
    time_values=dev_df["vintage_month"].to_numpy(),
)

# 3. Visually choose feature count
plot_rfe_curve(
    selector.results_,
    baseline_auc=selector.baseline_result_["mean_auc"],
    target_budget=50,
    zoom_max_features=120,
    save_path="rfe_curve.png",
)
plot_fold_stability(selector.results_, zoom_max_features=120, target_budget=50, save_path="fold_stability.png")

# 4. Extract exact feature list for your chosen count
chosen_features = selector.get_features(50)
plot_top_features(selector.get_feature_importances(50), top_n=50, save_path="top_50.png")
```

Run the fast local synthetic demo and test suite:

```bash
python examples/sagemaker_usage.py --synthetic --output-dir output_demo
pytest -v
```

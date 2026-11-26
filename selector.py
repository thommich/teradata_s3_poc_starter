"""Three-stage coarse-to-fine LightGBM Recursive Feature Elimination selector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import time
from typing import Any, Literal, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .data import PreparedMatrix, extract_step_matrix, prepare_matrix
from .splits import stratified_time_subsample, validate_cv_splits


RankingMode = Literal["val_shap_stability", "val_shap_mean", "gain"]
Stage1Method = Literal["geometric_shap_rfe", "null_importance"]


@dataclass
class StagedRFEConfig:
    """Configuration for 3-stage coarse-to-fine LightGBM RFE."""

    stage1_method: Stage1Method = "geometric_shap_rfe"
    stage1_target: int = 300
    stage1_drop_fraction: float = 0.25
    stage1_max_train_rows: int = 300_000
    stage1_max_bin: int = 127
    stage1_colsample_bytree: float = 0.25
    null_importance_runs: int = 20

    stage2_target: int = 70
    stage2_drop_fraction: float = 0.15
    stage2_max_bin: int = 255
    stage2_colsample_bytree: float = 0.50

    stage3_step: int = 1
    stage3_colsample_bytree: float = 0.80
    min_features: int = 10

    ranking_mode: RankingMode = "val_shap_stability"
    stability_penalty: float = 0.25
    shap_max_val_rows: int = 50_000
    eval_full_baseline: bool = True
    collinearity_threshold: float | None = None
    collinearity_sample_rows: int = 50_000

    n_jobs: int = -1
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    lgb_params: dict[str, Any] | None = None
    random_state: int = 42
    checkpoint_dir: str | Path | None = None
    verbose: bool = True


_DEFAULT_LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "verbose": -1,
}


def _next_n_features(current_n: int, target_n: int, drop_fraction: float, min_step: int = 1) -> int:
    """Compute strictly decreasing next feature count clamped to ``target_n``."""
    n_drop = max(min_step, int(np.floor(current_n * drop_fraction)))
    return max(target_n, current_n - n_drop)


def _stratified_val_subset(
    y_val: np.ndarray,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    """Select up to ``max_rows`` validation row indices stratified by binary target."""
    n = len(y_val)
    if max_rows <= 0 or n <= max_rows:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    pos_loc = np.flatnonzero(y_val == 1)
    neg_loc = np.flatnonzero(y_val != 1)
    pos_take = min(len(pos_loc), max(1, int(round(max_rows * (len(pos_loc) / n)))))
    neg_take = min(len(neg_loc), max_rows - pos_take)
    picked_pos = rng.choice(pos_loc, size=pos_take, replace=False) if pos_take < len(pos_loc) else pos_loc
    picked_neg = rng.choice(neg_loc, size=neg_take, replace=False) if neg_take < len(neg_loc) else neg_loc
    idx = np.concatenate([picked_pos, picked_neg])
    idx.sort()
    return idx


def _permute_within_time(
    y_tr: np.ndarray,
    time_tr: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute binary target within each time period to preserve temporal base-rate drift."""
    if time_tr is None:
        return rng.permutation(y_tr)
    y_perm = y_tr.copy()
    _, inverse = np.unique(time_tr, return_inverse=True)
    for p_id in range(int(inverse.max()) + 1):
        loc = np.flatnonzero(inverse == p_id)
        if len(loc) > 1:
            y_perm[loc] = rng.permutation(y_perm[loc])
    return y_perm


class StagedLGBMRFE:
    """Memory-safe 3-stage LightGBM Recursive Feature Elimination with expanding time CV."""

    def __init__(self, config: StagedRFEConfig | None = None) -> None:
        self.config = config or StagedRFEConfig()
        self.results_: pd.DataFrame = pd.DataFrame()
        self.baseline_result_: dict[str, Any] | None = None
        self.feature_sets_: dict[int, list[str]] = {}
        self.feature_importances_: dict[int, pd.DataFrame] = {}
        self._history_rows: list[dict[str, Any]] = []
        self._feature_name_to_idx: dict[str, int] = {}
        self._n_splits: int = 0

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg, flush=True)

    def _build_lgb_params(
        self,
        *,
        colsample_bytree: float,
        max_bin: int,
        seed_offset: int = 0,
    ) -> dict[str, Any]:
        params = dict(_DEFAULT_LGB_PARAMS)
        if self.config.lgb_params:
            params.update(self.config.lgb_params)
        params["colsample_bytree"] = colsample_bytree
        params["max_bin"] = max_bin
        params["n_jobs"] = self.config.n_jobs
        params["seed"] = self.config.random_state + seed_offset
        params["verbose"] = -1
        return params

    def _evaluate_feature_subset(
        self,
        prepared: PreparedMatrix,
        y: np.ndarray,
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
        active_indices: np.ndarray,
        *,
        stage_name: str,
        step_num: int,
        colsample_bytree: float,
        max_bin: int,
        max_train_rows: int | None,
        time_values: np.ndarray | None,
        ranking_mode: RankingMode | None = None,
        compute_importance: bool = True,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Evaluate a single active feature subset sequentially across all expanding CV folds."""
        t0 = time.perf_counter()
        mode = ranking_mode or self.config.ranking_mode
        n_features = len(active_indices)
        feature_names = [prepared.feature_names[i] for i in active_indices.tolist()]
        X_step, local_cat = extract_step_matrix(prepared, active_indices)

        fold_aucs: list[float] = []
        fold_best_iters: list[int] = []
        fold_train_rows: list[int] = []
        fold_norm_imps: list[np.ndarray] = []

        for fold_id, (tr_idx, va_idx) in enumerate(cv_splits):
            if max_train_rows is not None and len(tr_idx) > max_train_rows:
                sub_tr_idx, sample_w = stratified_time_subsample(
                    tr_idx,
                    y,
                    time_values=time_values,
                    max_rows=max_train_rows,
                    keep_all_positives=True,
                    random_state=self.config.random_state + fold_id,
                )
            else:
                sub_tr_idx, sample_w = tr_idx, None

            X_tr = X_step[sub_tr_idx]
            y_tr = y[sub_tr_idx]
            X_va = X_step[va_idx]
            y_va = y[va_idx]
            fold_train_rows.append(len(sub_tr_idx))

            cat_arg: list[int] | str = local_cat if local_cat else "auto"
            dtrain = lgb.Dataset(
                X_tr,
                label=y_tr,
                weight=sample_w,
                categorical_feature=cat_arg,
                free_raw_data=True,
            )
            dval = lgb.Dataset(
                X_va,
                label=y_va,
                reference=dtrain,
                categorical_feature=cat_arg,
                free_raw_data=True,
            )

            params = self._build_lgb_params(
                colsample_bytree=colsample_bytree,
                max_bin=max_bin,
                seed_offset=fold_id,
            )
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=self.config.num_boost_round,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            best_iter = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else self.config.num_boost_round
            preds = booster.predict(X_va, num_iteration=best_iter)
            fold_aucs.append(float(roc_auc_score(y_va, preds)))
            fold_best_iters.append(int(best_iter))

            if compute_importance:
                if mode in ("val_shap_stability", "val_shap_mean"):
                    shap_loc = _stratified_val_subset(
                        y_va,
                        self.config.shap_max_val_rows,
                        seed=self.config.random_state + 1000 + fold_id,
                    )
                    shap_vals = booster.predict(
                        X_va[shap_loc],
                        pred_contrib=True,
                        num_iteration=best_iter,
                    )
                    if not isinstance(shap_vals, np.ndarray):
                        shap_vals = shap_vals.toarray()
                    # Binary TreeSHAP returns (n_samples, n_features + 1); last column is bias
                    raw_imp = np.abs(shap_vals[:, :n_features]).mean(axis=0, dtype=np.float64)
                else:
                    raw_imp = np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float64)

                total = float(raw_imp.sum())
                norm_imp = raw_imp / total if total > 0.0 else np.full(n_features, 1.0 / n_features, dtype=np.float64)
                fold_norm_imps.append(norm_imp)

            del dtrain, dval, booster, X_tr, X_va

        del X_step
        if n_features > 500:
            gc.collect()

        aucs_arr = np.asarray(fold_aucs, dtype=np.float64)
        mean_auc = float(aucs_arr.mean())
        std_auc = float(aucs_arr.std(ddof=1)) if len(aucs_arr) > 1 else 0.0
        se_auc = std_auc / np.sqrt(len(aucs_arr))
        elapsed = time.perf_counter() - t0

        row: dict[str, Any] = {
            "step": step_num,
            "stage": stage_name,
            "is_full_data": max_train_rows is None,
            "n_features": n_features,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "se_auc": float(se_auc),
            "min_auc": float(aucs_arr.min()),
            "max_auc": float(aucs_arr.max()),
            "mean_best_iteration": float(np.mean(fold_best_iters)),
            "fit_seconds": float(elapsed),
            "train_rows_mean": float(np.mean(fold_train_rows)),
        }
        for k, auc_val in enumerate(fold_aucs):
            row[f"fold_{k}_auc"] = auc_val

        if compute_importance:
            imp_matrix = np.vstack(fold_norm_imps)  # shape (n_splits, n_features)
            mean_imp = imp_matrix.mean(axis=0)
            std_imp = imp_matrix.std(axis=0, ddof=1) if imp_matrix.shape[0] > 1 else np.zeros(n_features)
            if mode == "val_shap_stability":
                score = mean_imp - self.config.stability_penalty * std_imp
            else:
                score = mean_imp

            imp_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "score": score,
                    "mean_norm_imp": mean_imp,
                    "std_norm_imp": std_imp,
                }
            )
            for k in range(imp_matrix.shape[0]):
                imp_df[f"fold_{k}_imp"] = imp_matrix[k]
            imp_df = imp_df.sort_values(["score", "mean_norm_imp"], ascending=False).reset_index(drop=True)
        else:
            imp_df = pd.DataFrame(columns=["feature", "score", "mean_norm_imp", "std_norm_imp"])

        return row, imp_df

    def _run_null_importance_stage1(
        self,
        prepared: PreparedMatrix,
        y: np.ndarray,
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
        active_indices: np.ndarray,
        time_values: np.ndarray | None,
        step_num: int,
    ) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
        """Screen initial features using permuted-target Null Importance (PIMP)."""
        self._log(
            f"[Stage 1 | Null Importance] Fitting real baseline + {self.config.null_importance_runs} "
            f"target-permuted runs on {len(active_indices)} features..."
        )
        real_row, real_imp_df = self._evaluate_feature_subset(
            prepared,
            y,
            cv_splits,
            active_indices,
            stage_name="stage1_null_importance",
            step_num=step_num,
            colsample_bytree=self.config.stage1_colsample_bytree,
            max_bin=self.config.stage1_max_bin,
            max_train_rows=self.config.stage1_max_train_rows,
            time_values=time_values,
            ranking_mode="gain",
            compute_importance=True,
        )

        # Align real gain in active_indices order
        feature_names = [prepared.feature_names[i] for i in active_indices.tolist()]
        real_gain_map = dict(zip(real_imp_df["feature"], real_imp_df["mean_norm_imp"]))
        real_gain = np.asarray([real_gain_map[f] for f in feature_names], dtype=np.float64)

        X_step, local_cat = extract_step_matrix(prepared, active_indices)
        cat_arg: list[int] | str = local_cat if local_cat else "auto"
        n_features = len(active_indices)
        null_gains = np.zeros((self.config.null_importance_runs, n_features), dtype=np.float64)
        rng = np.random.default_rng(self.config.random_state + 9000)

        for run_idx in range(self.config.null_importance_runs):
            run_gain = np.zeros(n_features, dtype=np.float64)
            for fold_id, (tr_idx, va_idx) in enumerate(cv_splits):
                if len(tr_idx) > self.config.stage1_max_train_rows:
                    sub_tr_idx, sample_w = stratified_time_subsample(
                        tr_idx,
                        y,
                        time_values=time_values,
                        max_rows=self.config.stage1_max_train_rows,
                        keep_all_positives=True,
                        random_state=self.config.random_state + run_idx * 100 + fold_id,
                    )
                else:
                    sub_tr_idx, sample_w = tr_idx, None

                time_tr = time_values[sub_tr_idx] if time_values is not None else None
                y_perm_tr = _permute_within_time(y[sub_tr_idx], time_tr, rng)

                dtrain = lgb.Dataset(
                    X_step[sub_tr_idx],
                    label=y_perm_tr,
                    weight=sample_w,
                    categorical_feature=cat_arg,
                    free_raw_data=True,
                )
                params = self._build_lgb_params(
                    colsample_bytree=self.config.stage1_colsample_bytree,
                    max_bin=self.config.stage1_max_bin,
                    seed_offset=1000 + run_idx * 10 + fold_id,
                )
                # Cap null-run rounds to keep screening fast
                null_rounds = max(50, min(250, int(round(real_row["mean_best_iteration"]))))
                booster = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=null_rounds,
                    callbacks=[lgb.log_evaluation(period=0)],
                )
                g = np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float64)
                total = float(g.sum())
                if total > 0.0:
                    g /= total
                run_gain += g / len(cv_splits)
                del dtrain, booster

            null_gains[run_idx] = run_gain

        del X_step
        gc.collect()

        p95_null = np.percentile(null_gains, 95, axis=0)
        eps = 1e-7
        pimp_score = np.log((real_gain + eps) / (p95_null + eps))

        pimp_df = pd.DataFrame(
            {
                "feature": feature_names,
                "score": pimp_score,
                "mean_norm_imp": real_gain,
                "std_norm_imp": real_imp_df.set_index("feature").loc[feature_names, "std_norm_imp"].to_numpy(),
                "null_p95_imp": p95_null,
            }
        ).sort_values(["score", "mean_norm_imp"], ascending=False).reset_index(drop=True)

        keep_n = min(self.config.stage1_target, len(active_indices))
        kept_names = pimp_df["feature"].iloc[:keep_n].tolist()
        kept_indices = np.asarray([self._feature_name_to_idx[f] for f in kept_names], dtype=np.int64)
        return real_row, pimp_df, kept_indices

    def _prune_collinear_features(
        self,
        prepared: PreparedMatrix,
        active_indices: np.ndarray,
        imp_df: pd.DataFrame,
        threshold: float,
    ) -> np.ndarray:
        """Keep the highest-importance representative from each tight Spearman cluster."""
        if len(active_indices) <= 2:
            return active_indices

        rng = np.random.default_rng(self.config.random_state + 77)
        n_rows = min(prepared.n_samples, self.config.collinearity_sample_rows)
        if n_rows < prepared.n_samples:
            row_idx = np.sort(rng.choice(prepared.n_samples, size=n_rows, replace=False))
        else:
            row_idx = np.arange(prepared.n_samples, dtype=np.int64)

        X_sample = prepared.X[row_idx][:, active_indices].astype(np.float64, copy=False)
        # Impute NaNs with column medians for Spearman calculation
        col_medians = np.nanmedian(X_sample, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        nan_mask = np.isnan(X_sample)
        if nan_mask.any():
            X_sample = X_sample.copy()
            X_sample[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

        corr, _ = spearmanr(X_sample, axis=0)
        corr = np.nan_to_num(np.abs(corr), nan=0.0)
        np.fill_diagonal(corr, 1.0)
        dist = np.clip(1.0 - corr, 0.0, 1.0)
        dist = 0.5 * (dist + dist.T)
        np.fill_diagonal(dist, 0.0)

        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        clusters = fcluster(Z, t=1.0 - threshold, criterion="distance")

        score_map = dict(zip(imp_df["feature"], imp_df["score"]))
        best_per_cluster: dict[int, tuple[float, int]] = {}
        for local_j, global_j in enumerate(active_indices.tolist()):
            c_id = int(clusters[local_j])
            fname = prepared.feature_names[global_j]
            fscore = float(score_map.get(fname, 0.0))
            if c_id not in best_per_cluster or fscore > best_per_cluster[c_id][0]:
                best_per_cluster[c_id] = (fscore, global_j)

        kept = np.asarray([pair[1] for pair in best_per_cluster.values()], dtype=np.int64)
        self._log(
            f"[Collinearity Filter] |Spearman| > {threshold:.2f}: "
            f"{len(active_indices)} -> {len(kept)} cluster representatives."
        )
        return kept

    def _record_step(
        self,
        row: dict[str, Any],
        imp_df: pd.DataFrame,
        active_features: list[str],
    ) -> None:
        n_feat = int(row["n_features"])
        self._history_rows.append(row)
        self.feature_sets_[n_feat] = list(active_features)
        self.feature_importances_[n_feat] = imp_df

    def _save_checkpoint(
        self,
        active_features: list[str],
        stage_label: str,
    ) -> None:
        if not self.config.checkpoint_dir:
            return
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(self._history_rows).to_csv(ckpt_dir / "history.csv", index=False)

        feature_payload = {
            str(n_feat): {
                "features": feats,
                "importances": self.feature_importances_[n_feat].to_dict(orient="records")
                if n_feat in self.feature_importances_
                else [],
            }
            for n_feat, feats in self.feature_sets_.items()
        }
        (ckpt_dir / "feature_sets.json").write_text(json.dumps(feature_payload, indent=2), encoding="utf-8")

        state_payload = {
            "stage": stage_label,
            "active_features": active_features,
            "baseline_result": self.baseline_result_,
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self.config).items()},
        }
        (ckpt_dir / "state.json").write_text(json.dumps(state_payload, indent=2), encoding="utf-8")

    def _load_checkpoint(self) -> tuple[list[str], str] | None:
        if not self.config.checkpoint_dir:
            return None
        ckpt_dir = Path(self.config.checkpoint_dir)
        state_path = ckpt_dir / "state.json"
        history_path = ckpt_dir / "history.csv"
        fsets_path = ckpt_dir / "feature_sets.json"
        if not (state_path.exists() and history_path.exists() and fsets_path.exists()):
            return None

        state = json.loads(state_path.read_text(encoding="utf-8"))
        hist_df = pd.read_csv(history_path)
        self._history_rows = hist_df.to_dict(orient="records")
        self.baseline_result_ = state.get("baseline_result")

        fsets_raw = json.loads(fsets_path.read_text(encoding="utf-8"))
        self.feature_sets_ = {}
        self.feature_importances_ = {}
        for k_str, entry in fsets_raw.items():
            k = int(k_str)
            self.feature_sets_[k] = list(entry["features"])
            self.feature_importances_[k] = pd.DataFrame(entry.get("importances", []))

        return list(state["active_features"]), str(state["stage"])

    def fit(
        self,
        X: pd.DataFrame | PreparedMatrix,
        y: pd.Series | np.ndarray,
        cv_splits: Sequence[tuple[np.ndarray, np.ndarray]],
        time_values: pd.Series | pd.Index | np.ndarray | None = None,
        feature_cols: Sequence[str] | None = None,
        categorical_cols: Sequence[str] | None = None,
        resume: bool = False,
    ) -> StagedLGBMRFE:
        """Run the 3-stage coarse-to-fine LightGBM RFE pipeline."""
        if isinstance(X, PreparedMatrix):
            prepared = X
        else:
            prepared = prepare_matrix(X, feature_cols=feature_cols, categorical_cols=categorical_cols)

        y_arr = np.asarray(y, dtype=np.int8).ravel()
        if len(y_arr) != prepared.n_samples:
            raise ValueError(f"y length ({len(y_arr)}) does not match X rows ({prepared.n_samples}).")

        time_arr = np.asarray(time_values) if time_values is not None else None
        if time_arr is not None and len(time_arr) != prepared.n_samples:
            raise ValueError(f"time_values length ({len(time_arr)}) does not match X rows ({prepared.n_samples}).")

        splits = validate_cv_splits(cv_splits, prepared.n_samples)
        self._n_splits = len(splits)
        self._feature_name_to_idx = {name: i for i, name in enumerate(prepared.feature_names)}

        self.results_ = pd.DataFrame()
        self.baseline_result_ = None
        self.feature_sets_ = {}
        self.feature_importances_ = {}
        self._history_rows = []

        active_indices = np.arange(prepared.n_features, dtype=np.int64)
        step_num = 0
        resumed_stage = None

        if resume:
            loaded = self._load_checkpoint()
            if loaded is not None:
                active_names, resumed_stage = loaded
                active_indices = np.asarray([self._feature_name_to_idx[f] for f in active_names], dtype=np.int64)
                step_num = len(self._history_rows)
                self._log(
                    f"[Resume] Loaded checkpoint at step={step_num}, stage={resumed_stage}, "
                    f"active_features={len(active_indices)}."
                )

        # Optional full-data baseline at initial feature count if Stage 1 subsamples
        need_stage1 = len(active_indices) > self.config.stage1_target
        max_train_len = max(len(tr) for tr, _ in splits)
        will_subsample_stage1 = need_stage1 and max_train_len > self.config.stage1_max_train_rows

        if self.config.eval_full_baseline and self.baseline_result_ is None and will_subsample_stage1:
            self._log(
                f"[Baseline] Evaluating all {len(active_indices)} features on 100% of rows "
                f"across {len(splits)} expanding folds..."
            )
            base_row, base_imp = self._evaluate_feature_subset(
                prepared,
                y_arr,
                splits,
                active_indices,
                stage_name="baseline_full",
                step_num=step_num,
                colsample_bytree=self.config.stage1_colsample_bytree,
                max_bin=self.config.stage1_max_bin,
                max_train_rows=None,
                time_values=time_arr,
                compute_importance=True,
            )
            self.baseline_result_ = base_row
            active_names = [prepared.feature_names[i] for i in active_indices.tolist()]
            self._record_step(base_row, base_imp, active_names)
            self._save_checkpoint(active_names, "baseline_full")
            step_num += 1
            self._log(
                f"[Baseline] n={base_row['n_features']} | mean_auc={base_row['mean_auc']:.5f} "
                f"(SE={base_row['se_auc']:.5f}) | {base_row['fit_seconds']:.1f}s"
            )

        # Stage 1: Coarse screening (N -> stage1_target)
        if len(active_indices) > self.config.stage1_target and resumed_stage != "done":
            if self.config.stage1_method == "null_importance":
                s1_row, s1_imp, kept_indices = self._run_null_importance_stage1(
                    prepared, y_arr, splits, active_indices, time_arr, step_num
                )
                active_names = [prepared.feature_names[i] for i in active_indices.tolist()]
                self._record_step(s1_row, s1_imp, active_names)
                if self.baseline_result_ is None and s1_row["is_full_data"]:
                    self.baseline_result_ = s1_row
                step_num += 1
                active_indices = kept_indices
                if self.config.collinearity_threshold is not None:
                    active_indices = self._prune_collinear_features(
                        prepared, active_indices, s1_imp, self.config.collinearity_threshold
                    )
                self._save_checkpoint(
                    [prepared.feature_names[i] for i in active_indices.tolist()],
                    "stage1_done",
                )
            else:
                last_imp_df: pd.DataFrame | None = None
                while len(active_indices) > self.config.stage1_target:
                    row, imp_df = self._evaluate_feature_subset(
                        prepared,
                        y_arr,
                        splits,
                        active_indices,
                        stage_name="stage1_coarse",
                        step_num=step_num,
                        colsample_bytree=self.config.stage1_colsample_bytree,
                        max_bin=self.config.stage1_max_bin,
                        max_train_rows=self.config.stage1_max_train_rows,
                        time_values=time_arr,
                    )
                    active_names = [prepared.feature_names[i] for i in active_indices.tolist()]
                    self._record_step(row, imp_df, active_names)
                    if self.baseline_result_ is None and row["is_full_data"]:
                        self.baseline_result_ = row
                    step_num += 1
                    last_imp_df = imp_df

                    next_n = _next_n_features(
                        len(active_indices),
                        self.config.stage1_target,
                        self.config.stage1_drop_fraction,
                    )
                    kept_names = imp_df["feature"].iloc[:next_n].tolist()
                    active_indices = np.asarray(
                        [self._feature_name_to_idx[f] for f in kept_names],
                        dtype=np.int64,
                    )
                    self._log(
                        f"[Stage 1] n={row['n_features']} -> {len(active_indices)} | "
                        f"mean_auc={row['mean_auc']:.5f} (SE={row['se_auc']:.5f}) | "
                        f"{row['fit_seconds']:.1f}s"
                    )
                    self._save_checkpoint(kept_names, "stage1_coarse")

                if self.config.collinearity_threshold is not None and last_imp_df is not None:
                    active_indices = self._prune_collinear_features(
                        prepared, active_indices, last_imp_df, self.config.collinearity_threshold
                    )
                    self._save_checkpoint(
                        [prepared.feature_names[i] for i in active_indices.tolist()],
                        "stage1_done",
                    )

        # Stage 2: Medium pruning on 100% of rows (stage1_target -> stage2_target)
        while len(active_indices) > self.config.stage2_target and resumed_stage != "done":
            row, imp_df = self._evaluate_feature_subset(
                prepared,
                y_arr,
                splits,
                active_indices,
                stage_name="stage2_medium",
                step_num=step_num,
                colsample_bytree=self.config.stage2_colsample_bytree,
                max_bin=self.config.stage2_max_bin,
                max_train_rows=None,
                time_values=time_arr,
            )
            active_names = [prepared.feature_names[i] for i in active_indices.tolist()]
            self._record_step(row, imp_df, active_names)
            if self.baseline_result_ is None:
                self.baseline_result_ = row
            step_num += 1

            next_n = _next_n_features(
                len(active_indices),
                self.config.stage2_target,
                self.config.stage2_drop_fraction,
            )
            kept_names = imp_df["feature"].iloc[:next_n].tolist()
            active_indices = np.asarray([self._feature_name_to_idx[f] for f in kept_names], dtype=np.int64)
            self._log(
                f"[Stage 2] n={row['n_features']} -> {len(active_indices)} | "
                f"mean_auc={row['mean_auc']:.5f} (SE={row['se_auc']:.5f}) | "
                f"{row['fit_seconds']:.1f}s"
            )
            self._save_checkpoint(kept_names, "stage2_medium")

        # Stage 3: Fine 1-by-1 RFE on 100% of rows (stage2_target -> min_features)
        while len(active_indices) >= self.config.min_features and resumed_stage != "done":
            n_cur = len(active_indices)
            already_evaluated = any(
                r["n_features"] == n_cur and r["stage"] == "stage3_fine" for r in self._history_rows
            )
            if already_evaluated:
                break

            row, imp_df = self._evaluate_feature_subset(
                prepared,
                y_arr,
                splits,
                active_indices,
                stage_name="stage3_fine",
                step_num=step_num,
                colsample_bytree=self.config.stage3_colsample_bytree,
                max_bin=self.config.stage2_max_bin,
                max_train_rows=None,
                time_values=time_arr,
            )
            active_names = [prepared.feature_names[i] for i in active_indices.tolist()]
            self._record_step(row, imp_df, active_names)
            if self.baseline_result_ is None:
                self.baseline_result_ = row
            step_num += 1

            self._log(
                f"[Stage 3] n={n_cur} | mean_auc={row['mean_auc']:.5f} "
                f"(SE={row['se_auc']:.5f}, min_fold={row['min_auc']:.5f}) | "
                f"{row['fit_seconds']:.1f}s"
            )

            if n_cur <= self.config.min_features:
                self._save_checkpoint(active_names, "done")
                break

            next_n = max(self.config.min_features, n_cur - max(1, self.config.stage3_step))
            kept_names = imp_df["feature"].iloc[:next_n].tolist()
            active_indices = np.asarray([self._feature_name_to_idx[f] for f in kept_names], dtype=np.int64)
            self._save_checkpoint(kept_names, "stage3_fine")

        self.results_ = pd.DataFrame(self._history_rows)
        return self

    def get_features(self, n_features: int, allow_nearest: bool = False) -> list[str]:
        """Return the exact list of selected features for an evaluated ``n_features``."""
        if not self.feature_sets_:
            raise RuntimeError("Selector has not been fitted yet.")
        if n_features in self.feature_sets_:
            return list(self.feature_sets_[n_features])
        if not allow_nearest:
            available = sorted(self.feature_sets_.keys())
            raise KeyError(
                f"n_features={n_features} was not directly evaluated. "
                f"Pass allow_nearest=True or choose from evaluated counts (e.g. min={available[0]}, max={available[-1]})."
            )
        nearest = min(self.feature_sets_.keys(), key=lambda k: (abs(k - n_features), -k))
        return list(self.feature_sets_[nearest])

    def get_feature_importances(self, n_features: int) -> pd.DataFrame:
        """Return per-feature cross-fold importance diagnostics at ``n_features``."""
        if n_features not in self.feature_importances_:
            raise KeyError(f"No importance table recorded for n_features={n_features}.")
        return self.feature_importances_[n_features].copy()

    def suggest_n_features(
        self,
        method: Literal["one_se", "delta_auc", "max"] = "one_se",
        min_delta_auc: float = 0.0005,
        max_features: int | None = None,
        full_data_only: bool = True,
    ) -> dict[str, Any]:
        """Suggest the smallest parsimonious feature count meeting the specified criterion."""
        if self.results_.empty:
            raise RuntimeError("Selector has not been fitted yet.")

        df = self.results_.copy()
        if full_data_only and df["is_full_data"].any():
            df = df[df["is_full_data"]]
        if max_features is not None:
            filtered = df[df["n_features"] <= max_features]
            if not filtered.empty:
                df = filtered

        best_idx = df["mean_auc"].idxmax()
        best_row = df.loc[best_idx]
        best_auc = float(best_row["mean_auc"])
        best_se = float(best_row["se_auc"])

        if method == "max":
            chosen = best_row
            threshold_auc = best_auc
        elif method == "one_se":
            threshold_auc = best_auc - best_se
            candidates = df[df["mean_auc"] >= threshold_auc].sort_values("n_features", ascending=True)
            chosen = candidates.iloc[0]
        elif method == "delta_auc":
            threshold_auc = best_auc - min_delta_auc
            candidates = df[df["mean_auc"] >= threshold_auc].sort_values("n_features", ascending=True)
            chosen = candidates.iloc[0]
        else:
            raise ValueError(f"Unknown suggestion method '{method}'.")

        chosen_n = int(chosen["n_features"])
        return {
            "method": method,
            "n_features": chosen_n,
            "mean_auc": float(chosen["mean_auc"]),
            "se_auc": float(chosen["se_auc"]),
            "best_auc": best_auc,
            "best_n_features": int(best_row["n_features"]),
            "threshold_auc": float(threshold_auc),
            "features": self.get_features(chosen_n),
        }

    def evaluate_oot(
        self,
        X_train: pd.DataFrame | PreparedMatrix,
        y_train: pd.Series | np.ndarray,
        X_oot: pd.DataFrame | PreparedMatrix,
        y_oot: pd.Series | np.ndarray,
        n_features_list: Sequence[int],
        categorical_cols: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Train full-data LightGBM models for candidate feature counts and score on untouched OOT."""
        if self.results_.empty:
            raise RuntimeError("Selector must be fitted before calling evaluate_oot.")

        y_tr = np.asarray(y_train, dtype=np.int8).ravel()
        y_te = np.asarray(y_oot, dtype=np.int8).ravel()

        rows: list[dict[str, Any]] = []
        for n_req in n_features_list:
            feats = self.get_features(n_req, allow_nearest=True)
            actual_n = len(feats)

            if isinstance(X_train, PreparedMatrix):
                tr_idx = np.asarray([X_train.feature_names.index(f) for f in feats], dtype=np.int64)
                X_tr_mat, cat_tr = extract_step_matrix(X_train, tr_idx)
            else:
                prep_tr = prepare_matrix(X_train, feature_cols=feats, categorical_cols=categorical_cols)
                X_tr_mat = np.ascontiguousarray(prep_tr.X)
                cat_tr = sorted(prep_tr.categorical_indices)

            if isinstance(X_oot, PreparedMatrix):
                te_idx = np.asarray([X_oot.feature_names.index(f) for f in feats], dtype=np.int64)
                X_te_mat, _ = extract_step_matrix(X_oot, te_idx)
            else:
                prep_te = prepare_matrix(X_oot, feature_cols=feats, categorical_cols=categorical_cols)
                X_te_mat = np.ascontiguousarray(prep_te.X)

            # Match CV-derived optimal tree count for leak-free OOT evaluation
            match_rows = self.results_[self.results_["n_features"] == actual_n]
            if not match_rows.empty:
                n_rounds = max(10, int(round(float(match_rows.iloc[-1]["mean_best_iteration"]))))
            else:
                n_rounds = max(10, int(round(float(self.results_["mean_best_iteration"].median()))))

            colsample = (
                self.config.stage3_colsample_bytree
                if actual_n <= self.config.stage2_target
                else self.config.stage2_colsample_bytree
            )
            params = self._build_lgb_params(
                colsample_bytree=colsample,
                max_bin=self.config.stage2_max_bin,
                seed_offset=5000 + actual_n,
            )
            cat_arg: list[int] | str = cat_tr if cat_tr else "auto"
            dtrain = lgb.Dataset(X_tr_mat, label=y_tr, categorical_feature=cat_arg, free_raw_data=True)
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=n_rounds,
                callbacks=[lgb.log_evaluation(period=0)],
            )
            preds = booster.predict(X_te_mat)
            oot_auc = float(roc_auc_score(y_te, preds))

            cv_mean_auc = float(match_rows.iloc[-1]["mean_auc"]) if not match_rows.empty else np.nan
            rows.append(
                {
                    "n_features": actual_n,
                    "cv_mean_auc": cv_mean_auc,
                    "oot_auc": oot_auc,
                    "num_boost_round": n_rounds,
                }
            )
            del dtrain, booster, X_tr_mat, X_te_mat

        return pd.DataFrame(rows)

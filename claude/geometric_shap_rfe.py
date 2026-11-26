"""Geometric SHAP-RFE: recursive feature elimination for wide, tall tabular data.

Every round re-fits LightGBM on each time fold, ranks the features by mean |SHAP|
on that fold's validation window, and drops a *share of the remaining* features
(set by a schedule). Rounds therefore get both fewer and cheaper as the feature
set shrinks.

Usage
-----
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)   # C-contiguous
    y = df[target].to_numpy()
    folds = [(train_idx, val_idx), ...]          # your expanding time windows
    folds = downsample_negatives(folds, y, ratio=8)
    res = geometric_shap_rfe(X, y, folds, feature_names=feature_cols)
    res.curve                                    # DataFrame: n_features, auc_mean, auc_se
    res.selected                                 # chosen feature names (1-SE rule)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

DEFAULT_PARAMS = dict(
    objective="binary",
    learning_rate=0.1,
    num_leaves=31,
    min_data_in_leaf=200,
    max_bin=63,
    bagging_fraction=0.5,
    bagging_freq=1,
    feature_fraction=1.0,
    lambda_l2=1.0,
    force_col_wise=True,
    verbose=-1,
    seed=42,
    deterministic=True,
)

# (threshold, drop fraction): while n_remaining > threshold, drop this share.
DEFAULT_SCHEDULE = ((1000, 0.30), (300, 0.15), (80, 0.06))


def downsample_negatives(folds, y, ratio=8, seed=0):
    """Keep every positive in each TRAIN window and sample at most ratio*positives negatives.
    Validation windows are left alone."""
    rng = np.random.default_rng(seed)
    out = []
    for tr, va in folds:
        tr = np.asarray(tr)
        pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
        k = min(len(neg), ratio * len(pos))
        out.append((np.sort(np.concatenate([pos, rng.choice(neg, k, replace=False)])), np.asarray(va)))
    return out


def _n_drop(k, schedule, fine_step):
    for threshold, frac in schedule:
        if k > threshold:
            return max(fine_step, int(k * frac))
    return fine_step


def _fit_fold(X, y, tr, va, feats, params, num_boost_round, early_stopping, shap_rows, seed):
    Xtr = X[np.ix_(tr, feats)]
    Xva = X[np.ix_(va, feats)]
    # params must reach the Dataset, otherwise binning ignores max_bin etc.
    dtr = lgb.Dataset(Xtr, y[tr], params=params, free_raw_data=True)
    dva = lgb.Dataset(Xva, y[va], reference=dtr)   # inherits dtr's bin mappers
    booster = lgb.train(
        params, dtr, num_boost_round,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
    )
    it = booster.best_iteration or None
    auc = roc_auc_score(y[va], booster.predict(Xva, num_iteration=it))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(va), min(shap_rows, len(va)), replace=False)
    contrib = booster.predict(Xva[idx], pred_contrib=True, num_iteration=it)[:, :-1]
    imp = np.abs(contrib).mean(axis=0)
    return auc, imp / max(imp.sum(), 1e-12), booster.best_iteration


@dataclass
class RFEResult:
    history: list = field(default_factory=list)   # dicts per round
    feature_names: np.ndarray | None = None
    selected_idx: np.ndarray | None = None

    @property
    def curve(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{k: v for k, v in h.items() if k != "features"} for h in self.history]
        )

    @property
    def selected(self):
        return list(self.feature_names[self.selected_idx])

    def select(self, rule="1se", tol=0.001):
        """Choose the smallest set with AUC >= best - 1 SE ('1se') or >= best - tol ('tol')."""
        c = self.curve
        best = c["auc_mean"].idxmax()
        floor = c.loc[best, "auc_mean"] - (c.loc[best, "auc_se"] if rule == "1se" else tol)
        ok = c[c["auc_mean"] >= floor]
        pick = ok["n_features"].idxmin()
        self.selected_idx = self.history[pick]["features"]
        return self.selected


def geometric_shap_rfe(
    X, y, folds, feature_names=None, params=None, schedule=DEFAULT_SCHEDULE,
    fine_step=1, min_features=10, num_boost_round=5000, early_stopping=50,
    shap_rows=50_000, n_jobs=1, total_threads=None, start_features=None, verbose=True,
):
    """Run geometric SHAP-RFE. `folds` is a list of (train_idx, val_idx) arrays.

    n_jobs=1 (default) runs the folds one after another, each using every core.
    That was ~5x faster than parallel threads on Windows (OpenMP contention).
    n_jobs>1 uses worker processes (loky), and X is memmapped to disk for the workers.
    """
    import os
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.asarray(y)
    feature_names = np.asarray(feature_names if feature_names is not None
                               else [f"f{i}" for i in range(X.shape[1])])
    total_threads = total_threads or os.cpu_count()
    p = {**DEFAULT_PARAMS, **(params or {}), "num_threads": max(1, total_threads // n_jobs)}

    feats = np.arange(X.shape[1]) if start_features is None else np.sort(np.asarray(start_features))
    res = RFEResult(feature_names=feature_names)
    with Parallel(n_jobs=n_jobs, backend="loky" if n_jobs > 1 else "sequential") as pool:
        while len(feats) >= min_features:
            t0 = time.perf_counter()
            out = pool(
                delayed(_fit_fold)(X, y, tr, va, feats, p, num_boost_round,
                                   early_stopping, shap_rows, i)
                for i, (tr, va) in enumerate(folds)
            )
            aucs, imps, iters = zip(*out)
            aucs, imp = np.array(aucs), np.mean(imps, axis=0)
            res.history.append(dict(
                n_features=len(feats), auc_mean=aucs.mean(),
                auc_se=aucs.std(ddof=1) / np.sqrt(len(aucs)) if len(aucs) > 1 else 0.0,
                auc_folds=np.round(aucs, 5).tolist(), mean_trees=float(np.mean(iters)),
                seconds=round(time.perf_counter() - t0, 1), features=feats.copy(),
            ))
            if verbose:
                h = res.history[-1]
                print(f"[{len(res.history):3d}] k={h['n_features']:5d}  AUC={h['auc_mean']:.5f}"
                      f" ±{h['auc_se']:.5f}  trees={h['mean_trees']:.0f}  {h['seconds']}s", flush=True)
            n = max(_n_drop(len(feats), schedule, fine_step), int((imp == 0).sum()))
            n = min(n, len(feats) - min_features) if len(feats) > min_features else n
            if n <= 0:
                break
            feats = np.sort(feats[np.argsort(imp, kind="stable")[n:]])
    res.select()
    return res

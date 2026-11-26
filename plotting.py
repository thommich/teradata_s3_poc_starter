"""Visual diagnostic charts for selecting the optimal feature count after StagedLGBMRFE."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_STAGE_COLORS = {
    "baseline_full": "#475569",
    "stage1_coarse": "#f59e0b",
    "stage1_null_importance": "#f59e0b",
    "stage2_medium": "#2563eb",
    "stage3_fine": "#059669",
}

_STAGE_LABELS = {
    "baseline_full": "Full Baseline (100% rows)",
    "stage1_coarse": "Stage 1: Coarse (subsampled)",
    "stage1_null_importance": "Stage 1: Null Importance (subsampled)",
    "stage2_medium": "Stage 2: Medium (100% rows)",
    "stage3_fine": "Stage 3: Fine 1-by-1 (100% rows)",
}


def _suggest_from_df(
    df: pd.DataFrame,
    min_delta_auc: float,
) -> tuple[int, int, int]:
    """Return (peak_n, one_se_n, delta_n) from full-data RFE rows."""
    peak_row = df.loc[df["mean_auc"].idxmax()]
    peak_n = int(peak_row["n_features"])
    peak_auc = float(peak_row["mean_auc"])
    peak_se = float(peak_row["se_auc"])

    one_se_cands = df[df["mean_auc"] >= (peak_auc - peak_se)].sort_values("n_features")
    one_se_n = int(one_se_cands.iloc[0]["n_features"])

    delta_cands = df[df["mean_auc"] >= (peak_auc - min_delta_auc)].sort_values("n_features")
    delta_n = int(delta_cands.iloc[0]["n_features"])
    return peak_n, one_se_n, delta_n


def plot_rfe_curve(
    results_df: pd.DataFrame,
    baseline_auc: float | None = None,
    target_budget: int = 50,
    zoom_max_features: int = 120,
    min_delta_auc: float = 0.0005,
    show_fold_band: Literal["se", "std", "none"] = "se",
    invert_x: bool = False,
    figsize: tuple[float, float] = (15.5, 6.0),
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot full pruning trajectory (log scale) alongside a zoomed decision panel.

    Parameters
    ----------
    results_df:
        ``selector.results_`` DataFrame.
    baseline_auc:
        Optional horizontal reference line for the all-features full-data CV AUC.
    target_budget:
        Target feature budget to highlight (e.g., 50).
    zoom_max_features:
        Upper x-axis bound for the right-hand decision zoom panel.
    min_delta_auc:
        Tolerance below peak full-data AUC for the parsimony marker.
    show_fold_band:
        Shaded uncertainty band: ``"se"`` (±1 SE), ``"std"`` (±1 SD), or ``"none"``.
    invert_x:
        If True, reverses x-axis (large feature count on left -> small on right).
    figsize:
        Matplotlib figure size.
    save_path:
        Optional file path to save the figure (PNG/PDF/SVG).
    """
    if results_df.empty:
        raise ValueError("results_df is empty.")

    df = results_df.sort_values("n_features", ascending=True).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Left Panel: Full Trajectory (log-x) ---
    ax_full = axes[0]
    for stage_name, grp in df.groupby("stage", sort=False):
        grp_s = grp.sort_values("n_features")
        color = _STAGE_COLORS.get(str(stage_name), "#334155")
        label = _STAGE_LABELS.get(str(stage_name), str(stage_name))
        ax_full.plot(
            grp_s["n_features"],
            grp_s["mean_auc"],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=label,
        )
        if show_fold_band != "none":
            band_col = "se_auc" if show_fold_band == "se" else "std_auc"
            ax_full.fill_between(
                grp_s["n_features"],
                grp_s["mean_auc"] - grp_s[band_col],
                grp_s["mean_auc"] + grp_s[band_col],
                color=color,
                alpha=0.14,
            )

    if baseline_auc is not None:
        ax_full.axhline(
            baseline_auc,
            color="#0f172a",
            linestyle="--",
            linewidth=1.3,
            label=f"All-Feature Baseline ({baseline_auc:.4f})",
        )
    ax_full.axvline(
        target_budget,
        color="#dc2626",
        linestyle=":",
        linewidth=1.4,
        label=f"Target Budget (n={target_budget})",
    )
    ax_full.set_xscale("log")
    if invert_x:
        ax_full.invert_xaxis()
    ax_full.set_title("Full Staged RFE Trajectory (Log Feature Scale)", fontweight="bold")
    ax_full.set_xlabel("Number of Features (log scale)")
    ax_full.set_ylabel("Mean Expanding-CV ROC AUC")
    ax_full.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_full.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    # --- Right Panel: Decision Zone Zoom (linear-x, full-data rows) ---
    ax_zoom = axes[1]
    full_df = df[df["is_full_data"]] if df["is_full_data"].any() else df
    zoom_df = full_df[full_df["n_features"] <= zoom_max_features].sort_values("n_features")
    if len(zoom_df) < 2:
        zoom_df = full_df.sort_values("n_features")

    ax_zoom.plot(
        zoom_df["n_features"],
        zoom_df["mean_auc"],
        color="#059669",
        marker="o",
        markersize=4.5,
        linewidth=2.0,
        label="Mean CV AUC (100% rows)",
    )
    if show_fold_band != "none":
        band_col = "se_auc" if show_fold_band == "se" else "std_auc"
        band_label = "±1 SE across folds" if show_fold_band == "se" else "±1 SD across folds"
        ax_zoom.fill_between(
            zoom_df["n_features"],
            zoom_df["mean_auc"] - zoom_df[band_col],
            zoom_df["mean_auc"] + zoom_df[band_col],
            color="#059669",
            alpha=0.16,
            label=band_label,
        )

    peak_n, one_se_n, delta_n = _suggest_from_df(zoom_df, min_delta_auc=min_delta_auc)
    peak_auc = float(zoom_df.loc[zoom_df["n_features"] == peak_n, "mean_auc"].iloc[0])

    ax_zoom.scatter(
        [peak_n],
        [peak_auc],
        color="#1d4ed8",
        s=110,
        zorder=5,
        marker="*",
        label=f"Peak AUC: n={peak_n} ({peak_auc:.4f})",
    )
    ax_zoom.axvline(
        one_se_n,
        color="#7c3aed",
        linestyle="-.",
        linewidth=1.5,
        label=f"1-SE Rule: n={one_se_n}",
    )
    ax_zoom.axvline(
        delta_n,
        color="#0891b2",
        linestyle="--",
        linewidth=1.5,
        label=f"Within ΔAUC≤{min_delta_auc:.4f}: n={delta_n}",
    )

    budget_match = zoom_df[zoom_df["n_features"] == target_budget]
    if not budget_match.empty:
        b_auc = float(budget_match["mean_auc"].iloc[0])
        b_label = f"Budget n={target_budget} ({b_auc:.4f}, Δ={b_auc - peak_auc:+.4f})"
        ax_zoom.scatter([target_budget], [b_auc], color="#dc2626", s=55, zorder=6)
    else:
        b_label = f"Target Budget: n={target_budget}"
    ax_zoom.axvline(target_budget, color="#dc2626", linestyle=":", linewidth=1.6, label=b_label)

    if baseline_auc is not None:
        ax_zoom.axhline(
            baseline_auc,
            color="#0f172a",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            label=f"Full Baseline ({baseline_auc:.4f})",
        )

    if invert_x:
        ax_zoom.invert_xaxis()
    ax_zoom.set_title("Decision Zone Zoom (Full-Data Expanding CV)", fontweight="bold")
    ax_zoom.set_xlabel("Number of Features")
    ax_zoom.set_ylabel("Mean Expanding-CV ROC AUC")
    ax_zoom.grid(True, linestyle="--", alpha=0.35)
    ax_zoom.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    fig.tight_layout()
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig, axes


def plot_fold_stability(
    results_df: pd.DataFrame,
    zoom_max_features: int = 120,
    target_budget: int = 50,
    figsize: tuple[float, float] = (14.5, 5.2),
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot per-fold ROC AUC trajectories and temporal spread across expanding folds."""
    if results_df.empty:
        raise ValueError("results_df is empty.")

    df = results_df[results_df["is_full_data"]] if results_df["is_full_data"].any() else results_df
    zoom_df = df[df["n_features"] <= zoom_max_features].sort_values("n_features")
    if len(zoom_df) < 2:
        zoom_df = df.sort_values("n_features")

    fold_cols = sorted(
        [c for c in zoom_df.columns if c.startswith("fold_") and c.endswith("_auc")],
        key=lambda c: int(c.split("_")[1]),
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax_folds = axes[0]
    for col in fold_cols:
        fold_idx = col.split("_")[1]
        ax_folds.plot(
            zoom_df["n_features"],
            zoom_df[col],
            linewidth=1.3,
            alpha=0.75,
            label=f"Fold {fold_idx}",
        )
    ax_folds.plot(
        zoom_df["n_features"],
        zoom_df["mean_auc"],
        color="#0f172a",
        linewidth=2.3,
        label="Mean AUC",
    )
    ax_folds.axvline(target_budget, color="#dc2626", linestyle=":", linewidth=1.5, label=f"n={target_budget}")
    ax_folds.set_title("Per-Fold Expanding Time-Window AUC", fontweight="bold")
    ax_folds.set_xlabel("Number of Features")
    ax_folds.set_ylabel("Validation ROC AUC")
    ax_folds.grid(True, linestyle="--", alpha=0.35)
    ax_folds.legend(loc="lower right", fontsize=8.5, ncol=2)

    ax_stab = axes[1]
    line1 = ax_stab.plot(
        zoom_df["n_features"],
        zoom_df["min_auc"],
        color="#059669",
        marker="o",
        markersize=3.5,
        linewidth=1.9,
        label="Worst-Fold AUC (min_auc)",
    )
    ax_stab.set_xlabel("Number of Features")
    ax_stab.set_ylabel("Worst-Fold ROC AUC", color="#059669")
    ax_stab.tick_params(axis="y", labelcolor="#059669")
    ax_stab.axvline(target_budget, color="#dc2626", linestyle=":", linewidth=1.5)
    ax_stab.grid(True, linestyle="--", alpha=0.35)

    ax_std = ax_stab.twinx()
    line2 = ax_std.plot(
        zoom_df["n_features"],
        zoom_df["std_auc"],
        color="#d97706",
        linestyle="--",
        marker="s",
        markersize=3.2,
        linewidth=1.6,
        label="Cross-Fold Spread (std_auc)",
    )
    ax_std.set_ylabel("Cross-Fold AUC Std Dev", color="#d97706")
    ax_std.tick_params(axis="y", labelcolor="#d97706")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_stab.legend(lines, labels, loc="upper right", fontsize=8.5)
    ax_stab.set_title("Worst-Fold Robustness & Cross-Fold Spread", fontweight="bold")

    fig.tight_layout()
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig, axes


def plot_top_features(
    importance_df: pd.DataFrame,
    top_n: int = 50,
    figsize: tuple[float, float] = (10.0, 11.5),
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot horizontal bars of normalized validation importance with ±1 SD fold error bars."""
    if importance_df.empty:
        raise ValueError("importance_df is empty.")

    sub = importance_df.sort_values("mean_norm_imp", ascending=False).head(top_n).iloc[::-1]
    height = max(4.5, min(18.0, 0.24 * len(sub) + 1.5))
    fig, ax = plt.subplots(figsize=(figsize[0], height if figsize[1] == 11.5 else figsize[1]))

    y_pos = np.arange(len(sub))
    ax.barh(
        y_pos,
        sub["mean_norm_imp"],
        xerr=sub["std_norm_imp"],
        color="#2563eb",
        alpha=0.85,
        ecolor="#1e293b",
        capsize=2.5,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["feature"], fontsize=8.5)
    ax.set_xlabel("Mean Normalized Validation Importance (±1 SD across folds)")
    ax.set_title(f"Top {len(sub)} Selected Features — Cross-Fold Importance & Stability", fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig, ax

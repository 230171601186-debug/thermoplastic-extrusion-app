"""Metrics, algorithm selection, tables, and evaluation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from config import CV_FOLDS, RANDOM_STATE, RESULTS_DIR


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def format_metrics(metrics: dict[str, float]) -> str:
    return f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  R²={metrics['R2']:.4f}"


def print_metric_row(model_name: str, target: str, metrics: dict[str, float]) -> None:
    print(
        f"  {model_name:16}  target={target:32}  {format_metrics(metrics)}"
    )


def cv_metrics(pipeline, X, y, n_splits: int = CV_FOLDS) -> dict[str, float]:
    """Out-of-fold metrics on the training set only (test set stays untouched)."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    preds = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)
    return regression_metrics(y, preds), preds


def select_best_algorithm(cv_by_algo: dict[str, dict[str, float]]) -> str:
    """Pick the algorithm using MAE first, then RMSE, then R².

    Highest R² alone is not enough: on synthetic manufacturing data a model
    can look well-correlated while still being off on the process units that
    operators would actually set.
    """

    def sort_key(item):
        name, m = item
        return (m["MAE"], m["RMSE"], -m["R2"])

    ranked = sorted(cv_by_algo.items(), key=sort_key)
    return ranked[0][0]


def print_comparison_tables(target: str, results: dict[str, dict[str, float]], split_label: str) -> None:
    algos = list(results.keys())
    print(f"\n  [{split_label}] {target}")
    header = f"  {'Algorithm':16} {'MAE':>12} {'RMSE':>12} {'R²':>10}"
    print(header)
    print("  " + "-" * 52)
    for algo in algos:
        m = results[algo]
        print(f"  {algo:16} {m['MAE']:12.4f} {m['RMSE']:12.4f} {m['R2']:10.4f}")


def print_wide_r2_table(per_target: dict[str, dict[str, dict[str, float]]], metric: str, split_label: str) -> None:
    algos = next(iter(per_target.values())).keys()
    algo_list = list(algos)
    print(f"\n{'=' * 78}")
    print(f"MODEL 1 COMPARISON — {metric} ({split_label})")
    print("=" * 78)
    col_w = 14
    header = f"{'Target':32}" + "".join(f"{a:>{col_w}}" for a in algo_list)
    print(header)
    print("-" * len(header))
    key = {"R2": "R2", "MAE": "MAE", "RMSE": "RMSE"}[metric]
    for target, by_algo in per_target.items():
        row = f"{target:32}"
        for algo in algo_list:
            row += f"{by_algo[algo][key]:{col_w}.4f}"
        print(row)


def _ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def plot_metric_bars(per_target: dict[str, dict[str, dict[str, float]]], metric: str, title: str, filename: str) -> None:
    rows = []
    for target, by_algo in per_target.items():
        for algo, m in by_algo.items():
            rows.append({"Target": target, "Algorithm": algo, metric: m[metric if metric != "R²" else "R2"]})
    df = pd.DataFrame(rows)
    value_col = "R2" if metric == "R²" else metric
    pivot = df.pivot(index="Target", columns="Algorithm", values=value_col)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(title="Algorithm")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path = _ensure_results_dir() / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_actual_vs_predicted(y_true, y_pred, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, alpha=0.35, s=12, edgecolors="none")
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    path = _ensure_results_dir() / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_residuals(y_true, y_pred, title: str, filename: str) -> None:
    resid = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(y_pred, resid, alpha=0.35, s=12, edgecolors="none")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (predicted − actual)")
    ax.set_title(title)
    fig.tight_layout()
    path = _ensure_results_dir() / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_feature_importance(names: list[str], importances: np.ndarray, title: str, filename: str, top_n: int = 12) -> None:
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(np.array(names)[order][::-1], importances[order][::-1])
    ax.set_xlabel("Feature importance")
    ax.set_title(title)
    fig.tight_layout()
    path = _ensure_results_dir() / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_algorithm_comparison(results: dict[str, dict[str, float]], title: str, filename: str) -> None:
    """Grouped MAE / RMSE / R2 bars for a single-target model comparison."""
    algos = list(results.keys())
    mae = [results[a]["MAE"] for a in algos]
    rmse = [results[a]["RMSE"] for a in algos]
    r2 = [results[a]["R2"] for a in algos]
    x = np.arange(len(algos))
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, values, ylabel in zip(
        axes,
        [mae, rmse, r2],
        ["MAE", "RMSE", "R2"],
    ):
        ax.bar(x, values, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=20)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    path = _ensure_results_dir() / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_actual_vs_predicted_grid(panels: list[tuple[str, np.ndarray, np.ndarray]], filename: str, suptitle: str) -> None:
    n = len(panels)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, y_true, y_pred) in zip(axes, panels):
        ax.scatter(y_true, y_pred, alpha=0.3, s=8, edgecolors="none")
        lo = min(np.min(y_true), np.min(y_pred))
        hi = max(np.max(y_true), np.max(y_pred))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Actual", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(suptitle)
    fig.tight_layout()
    path = _ensure_results_dir() / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plot] Saved {path}")

"""Evaluation, bootstrap CIs, comparison tables, and error analysis."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    positive_label: str = "positive",
) -> Dict[str, float]:
    """Compute a standard set of classification metrics.

    'unk' predictions are left as-is and will count as wrong for any
    label they do not match. ``unk_rate`` is tracked separately.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_positive": f1_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "unk_rate": float(np.mean(y_pred == "unk")),
    }


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------
def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 10_000,
    seed: int = 42,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Bootstrap percentile confidence interval for a paired-sample metric.

    Resamples indices with replacement; applies the same indices to both
    ``y_true`` and ``y_pred`` so the pairing is preserved.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        values[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(values, alpha))
    hi = float(np.quantile(values, 1.0 - alpha))
    return lo, hi


def _accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def _f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def comparison_table(
    results: List[Dict],
    n_boot: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a summary table with point estimates and bootstrap 95% CIs.

    Each entry in ``results`` must be a dict with keys:
        ``name``   -- display name of the method
        ``y_true`` -- iterable of true labels
        ``y_pred`` -- iterable of predicted labels
    """
    rows = []
    for r in results:
        y_true = np.asarray(r["y_true"])
        y_pred = np.asarray(r["y_pred"])
        m = compute_metrics(y_true, y_pred)
        acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, _accuracy, n_boot=n_boot, seed=seed)
        f1_lo, f1_hi = bootstrap_ci(y_true, y_pred, _f1_macro, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "metodo": r["name"],
                "accuracy": m["accuracy"],
                "accuracy_ci95": f"[{acc_lo:.3f}, {acc_hi:.3f}]",
                "f1_macro": m["f1_macro"],
                "f1_macro_ci95": f"[{f1_lo:.3f}, {f1_hi:.3f}]",
                "f1_positive": m["f1_positive"],
                "precision_macro": m["precision_macro"],
                "recall_macro": m["recall_macro"],
                "unk_rate": m["unk_rate"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Confusion matrix plots
# ---------------------------------------------------------------------------
def plot_confusion_matrices(
    results: List[Dict],
    labels: Sequence[str] = ("positive", "negative"),
    figsize_per_plot: Tuple[int, int] = (4, 4),
):
    """Side-by-side confusion-matrix heatmaps, one per result dict."""
    n = len(results)
    fig, axes = plt.subplots(
        1, n, figsize=(figsize_per_plot[0] * n, figsize_per_plot[1])
    )
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = confusion_matrix(r["y_true"], r["y_pred"], labels=list(labels))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            ax=ax,
        )
        ax.set_title(r["name"])
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Real")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------
def error_analysis(
    df: pd.DataFrame,
    text_col: str,
    y_true_col: str,
    pred_cols: Dict[str, str],
    n: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Return representative misclassification examples.

    ``pred_cols`` maps a short model name to the column that holds its
    predictions. Must contain exactly two entries (typically the baseline
    and the SLM) so we can categorize errors as tfidf-only, slm-only,
    or both-wrong.
    """
    if len(pred_cols) != 2:
        raise ValueError("error_analysis expects exactly two prediction columns")

    (name_a, col_a), (name_b, col_b) = pred_cols.items()
    correct_a = df[y_true_col] == df[col_a]
    correct_b = df[y_true_col] == df[col_b]

    categories = {
        f"{name_a}_correct_{name_b}_wrong": df[correct_a & ~correct_b],
        f"{name_b}_correct_{name_a}_wrong": df[~correct_a & correct_b],
        "both_wrong": df[~correct_a & ~correct_b],
    }

    rng = np.random.default_rng(seed)
    parts = []
    for cat_name, cat_df in categories.items():
        if len(cat_df) == 0:
            continue
        k = min(n, len(cat_df))
        idx = rng.choice(cat_df.index.to_numpy(), size=k, replace=False)
        sampled = cat_df.loc[idx].copy()
        sampled["error_type"] = cat_name
        parts.append(sampled[[text_col, y_true_col, col_a, col_b, "error_type"]])
    return pd.concat(parts) if parts else pd.DataFrame()

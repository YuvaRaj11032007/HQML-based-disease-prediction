"""
Evaluation utilities: metrics computation, threshold selection, statistical tests.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: True labels (0 or 1).
        y_prob: Predicted probabilities for class 1.
        threshold: Decision threshold.

    Returns:
        Dict with accuracy, sensitivity, specificity, auc.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class in y_true, AUC undefined.")
        auc = float("nan")
    else:
        auc = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = tp / max(tp + fn, 1)  # recall for positive class
    specificity = tn / max(tn + fp, 1)  # recall for negative class
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Find optimal decision threshold using Youden's J statistic on validation data.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.

    Returns:
        Optimal threshold.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    logger.info(
        f"Optimal threshold: {best_threshold:.4f} "
        f"(J={j_scores[best_idx]:.4f})"
    )

    return float(best_threshold)


def paired_wilcoxon_test(
    scores_a: List[float],
    scores_b: List[float],
    metric_name: str = "AUC",
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test between two models' fold scores.

    Args:
        scores_a: Scores from model A (e.g., hybrid).
        scores_b: Scores from model B (e.g., baseline).
        metric_name: Name of the metric being compared.
        alpha: Significance level.

    Returns:
        Dict with statistic, p_value, and significant flag.
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    # Need at least 6 pairs for Wilcoxon, and differences must not all be zero
    diffs = scores_a - scores_b
    if len(diffs) < 6 or np.all(diffs == 0):
        logger.warning(
            f"Cannot perform Wilcoxon test for {metric_name}: "
            f"need >= 6 pairs with non-zero differences. "
            f"Got {len(diffs)} pairs."
        )
        return {"statistic": float("nan"), "p_value": float("nan"), "significant": False}

    stat, p_val = wilcoxon(scores_a, scores_b, alternative="two-sided")

    result = {
        "statistic": stat,
        "p_value": p_val,
        "significant": p_val < alpha,
    }

    logger.info(
        f"Wilcoxon test ({metric_name}): stat={stat:.4f}, p={p_val:.4f}, "
        f"significant={'Yes' if result['significant'] else 'No'}"
    )

    return result


def aggregate_cv_results(
    fold_metrics: List[Dict[str, float]],
) -> Dict[str, str]:
    """Aggregate CV fold metrics into mean ± std strings.

    Args:
        fold_metrics: List of metric dicts from each fold.

    Returns:
        Dict with "metric_name": "mean ± std" strings.
    """
    df = pd.DataFrame(fold_metrics)
    result = {}
    for col in df.columns:
        values = df[col].dropna().values
        if len(values) > 0:
            result[col] = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
        else:
            result[col] = "N/A"
    return result


def format_results_table(
    all_results: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """Format results from all models into a comparison table.

    Args:
        all_results: {model_name: {metric: "mean ± std"}}.

    Returns:
        DataFrame with models as rows and metrics as columns.
    """
    rows = []
    for model_name, metrics in all_results.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    if "Model" in df.columns:
        df = df.set_index("Model")
    return df

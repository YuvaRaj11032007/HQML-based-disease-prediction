"""
Explainability: permutation importance, SHAP, and pathway enrichment.
"""

import logging
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

logger = logging.getLogger(__name__)


def permutation_importance_manual(
    model,
    X: np.ndarray,
    y: np.ndarray,
    gene_names: List[str],
    n_repeats: int = 30,
    seed: int = 42,
    is_torch: bool = False,
) -> pd.DataFrame:
    """Compute permutation importance using ROC-AUC (robust to class imbalance).

    Args:
        model: Trained model (sklearn or torch).
        X: Feature matrix.
        y: True labels.
        gene_names: Feature names.
        n_repeats: Number of permutation repeats.
        seed: Random seed.
        is_torch: Whether the model is a PyTorch model.

    Returns:
        DataFrame with gene importances based on drop in ROC-AUC.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(seed)

    def predict_prob(X_in):
        if is_torch:
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_in, dtype=torch.float32))
                return preds.numpy().flatten()
        else:
            return model.predict_proba(X_in)[:, 1]

    baseline_prob = predict_prob(X)
    baseline_score = roc_auc_score(y, baseline_prob)
    importances = []

    for i, gene in enumerate(gene_names):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, i] = rng.permutation(X_permuted[:, i])
            perm_prob = predict_prob(X_permuted)
            perm_score = roc_auc_score(y, perm_prob)
            scores.append(baseline_score - perm_score)
        importances.append(
            {
                "gene": gene,
                "importance_mean": float(np.mean(scores)),
                "importance_std": float(np.std(scores)),
            }
        )

    df = pd.DataFrame(importances).sort_values("importance_mean", ascending=False)
    return df


def compute_selection_stability(
    all_selected_gene_lists: List[List[str]],
    top_k: int = 8,
    output_csv: str = "results/selection_stability.csv",
    output_png: Optional[str] = "results/selection_stability.png",
) -> pd.DataFrame:
    """Compute and report selection stability for biomarker genes across folds.

    Counts how frequently each gene is selected in the top-k qubit set across
    all folds and random seeds.

    Args:
        all_selected_gene_lists: List of gene lists (one per fold/seed).
        top_k: Number of genes selected per fold (e.g. 8).
        output_csv: Path to save stability CSV.
        output_png: Path to save stability bar plot.

    Returns:
        DataFrame with gene, count, frequency, rank.
    """
    from collections import Counter

    total_folds = len(all_selected_gene_lists)
    all_genes = [g for sublist in all_selected_gene_lists for g in sublist]
    counts = Counter(all_genes)

    rows = []
    for gene, count in counts.most_common():
        rows.append({
            "gene": gene,
            "selection_count": count,
            "total_folds": total_folds,
            "selection_frequency": count / max(total_folds, 1),
        })

    df = pd.DataFrame(rows)
    df["rank"] = range(1, len(df) + 1)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved selection stability to {output_csv}")

    if output_png and len(df) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        top_plot = df.head(15).sort_values("selection_frequency", ascending=True)
        ax.barh(top_plot["gene"], top_plot["selection_frequency"] * 100, color="teal", alpha=0.85)
        ax.set_xlabel("Selection Frequency Across Folds (%)")
        ax.set_title(f"Gene Selection Stability (Top {top_k} Qubit Genes across {total_folds} Folds)")
        ax.set_xlim([0, 105])
        for i, (val, cnt) in enumerate(zip(top_plot["selection_frequency"] * 100, top_plot["selection_count"])):
            ax.text(val + 1, i, f"{cnt}/{total_folds} ({val:.1f}%)", va="center", fontsize=9)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_png), exist_ok=True)
        plt.savefig(output_png, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved selection stability plot to {output_png}")

    return df


def shap_explain_hybrid(
    model: torch.nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    gene_names: List[str],
    n_samples: int = 100,
) -> Optional[np.ndarray]:
    """Compute SHAP values for the hybrid quantum model.

    Uses KernelExplainer since the quantum model is a black box.

    Args:
        model: Trained hybrid model.
        X_train: Background data for SHAP.
        X_test: Data to explain.
        gene_names: Feature names.
        n_samples: Number of SHAP samples.

    Returns:
        SHAP values array or None if SHAP fails.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed, skipping SHAP analysis.")
        return None

    model.eval()

    def predict_fn(X_in):
        with torch.no_grad():
            preds = model(torch.tensor(X_in, dtype=torch.float32))
            return preds.numpy().flatten()

    # Use a subset for background
    bg_size = min(50, len(X_train))
    background = X_train[:bg_size]

    try:
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_test, nsamples=n_samples)
        return shap_values
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None


def pathway_enrichment(
    gene_names: List[str],
    gene_sets: List[str],
    output_dir: str = "results",
) -> Optional[pd.DataFrame]:
    """Run pathway enrichment analysis using gseapy/Enrichr.

    Args:
        gene_names: List of selected gene symbols.
        gene_sets: Enrichr gene set libraries to query.
        output_dir: Directory for results.

    Returns:
        DataFrame with enrichment results or None.
    """
    try:
        import gseapy as gp
    except ImportError:
        logger.warning("gseapy not installed, skipping pathway enrichment.")
        return None

    try:
        enr = gp.enrichr(
            gene_list=gene_names,
            gene_sets=gene_sets,
            organism="human",
            outdir=os.path.join(output_dir, "enrichr"),
            no_plot=True,
        )
        results = enr.results
        if results is not None and len(results) > 0:
            logger.info(f"Enrichment found {len(results)} terms.")
            return results
        else:
            logger.info("No significant enrichment terms found.")
            return None
    except Exception as e:
        logger.warning(f"Pathway enrichment failed: {e}")
        return None


def plot_permutation_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    output_path: str,
) -> None:
    """Plot permutation importance bar chart.

    Args:
        importance_df: DataFrame with gene, importance_mean, importance_std.
        model_name: Name of the model.
        output_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    df = importance_df.sort_values("importance_mean", ascending=True)

    ax.barh(
        df["gene"],
        df["importance_mean"],
        xerr=df["importance_std"],
        color="steelblue",
        alpha=0.8,
    )
    ax.set_xlabel("Importance (decrease in ROC-AUC)")
    ax.set_title(f"Permutation Importance - {model_name}")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved importance plot: {output_path}")


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    gene_names: List[str],
    output_path: str,
) -> None:
    """Plot SHAP summary plot.

    Args:
        shap_values: SHAP values array.
        X: Feature matrix.
        gene_names: Feature names.
        output_path: Path to save the plot.
    """
    try:
        import shap

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=gene_names,
            show=False,
            plot_type="bar",
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP plot: {output_path}")
    except Exception as e:
        logger.warning(f"SHAP plot failed: {e}")


def plot_roc_comparison(
    roc_data: Dict[str, Dict],
    output_path: str,
) -> None:
    """Plot ROC curves for all models.

    Args:
        roc_data: {model_name: {"fpr": array, "tpr": array, "auc": float}}
        output_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(roc_data)))
    for (name, data), color in zip(roc_data.items(), colors):
        ax.plot(
            data["fpr"],
            data["tpr"],
            label=f'{name} (AUC={data["auc"]:.3f})',
            color=color,
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC plot: {output_path}")


def plot_learning_curve(
    results: Dict[str, List[Dict]],
    fractions: List[float],
    output_path: str,
) -> None:
    """Plot learning curves for all models.

    Args:
        results: {model_name: [metrics_per_fraction]}
        fractions: Training data fractions.
        output_path: Output path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, metrics_list in results.items():
        aucs = [m.get("auc", 0) for m in metrics_list]
        ax.plot(fractions, aucs, "o-", label=model_name, linewidth=2, markersize=8)

    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("AUC")
    ax.set_title("Learning Curves")
    ax.legend()
    ax.set_xlim([0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved learning curve: {output_path}")

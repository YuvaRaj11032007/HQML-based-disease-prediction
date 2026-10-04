"""
Main training orchestrator for the Hybrid Quantum-Classical ML pipeline.

Entry point: python -m src.train --config config.yaml [--synthetic] [--fast]

Orchestrates the full pipeline:
1. Data loading (real TCGA-BRCA or synthetic)
2. Preprocessing and patient-level splitting
3. Cross-validation with feature selection per fold
4. Training hybrid quantum, classical baselines, and ablation model
5. Evaluation, robustness experiments, and explainability
6. Saving results
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

from src.data import load_data, load_probemap
from src.evaluate import (
    aggregate_cv_results,
    compute_metrics,
    find_optimal_threshold,
    format_results_table,
    paired_wilcoxon_test,
)
from src.explain import (
    pathway_enrichment,
    permutation_importance_manual,
    plot_learning_curve,
    plot_permutation_importance,
    plot_roc_comparison,
    plot_shap_summary,
    shap_explain_hybrid,
)
from src.features import (
    get_patient_cv_folds,
    map_ensembl_to_symbols,
    patient_level_split,
    remove_low_quality_genes,
    scale_features,
    select_features_fold,
)
from src.models_classical import (
    get_classical_predictions,
    train_random_forest,
    train_svm,
    train_xgboost,
)
from src.models_quantum import (
    ClassicalAblationModel,
    HybridQuantumModel,
    compute_class_weights,
    train_torch_model,
)

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_torch_predictions(
    model: torch.nn.Module, X: np.ndarray
) -> np.ndarray:
    """Get probabilities from a PyTorch model."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        preds = model(X_t).numpy().flatten()
    return preds


def run_cv_experiment(
    X_trainval: pd.DataFrame,
    y_trainval: pd.Series,
    pid_trainval: pd.Series,
    config: dict,
    n_folds: int = 5,
    seed: int = 42,
    fast: bool = False,
    n_final_genes: int = 8,
) -> Dict[str, Any]:
    """Run one complete CV experiment for all models.

    Args:
        X_trainval: Training+validation expression data.
        y_trainval: Labels.
        pid_trainval: Patient IDs.
        config: Full config dict.
        n_folds: Number of CV folds.
        seed: Random seed.
        fast: If True, use fewer epochs.
        n_final_genes: Number of genes to select.

    Returns:
        Dict with fold results for each model.
    """
    set_all_seeds(seed)

    folds = get_patient_cv_folds(X_trainval, y_trainval, pid_trainval, n_folds, seed)

    feat_cfg = config["features"]
    qm_cfg = config["quantum_model"]
    cl_cfg = config["classical_baselines"]

    max_epochs = config["training"]["fast_max_epochs"] if fast else qm_cfg["max_epochs"]

    all_fold_results = {
        "hybrid": [],
        "ablation": [],
        "svm_8": [],
        "rf_8": [],
        "xgb_8": [],
        "svm_50": [],
        "rf_50": [],
        "xgb_50": [],
    }
    all_selected_genes = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_idx + 1}/{n_folds} (seed={seed})")
        logger.info(f"{'='*60}")

        X_fold_train = X_trainval.iloc[train_idx]
        y_fold_train = y_trainval.iloc[train_idx]
        X_fold_val = X_trainval.iloc[val_idx]
        y_fold_val = y_trainval.iloc[val_idx]

        # --- Feature selection on training fold only ---
        selected_genes = select_features_fold(
            X_fold_train,
            y_fold_train,
            n_de_genes=feat_cfg["n_de_genes"],
            n_final_genes=n_final_genes,
            seed=seed,
        )
        all_selected_genes.append(selected_genes)

        # Also select top 50 for classical baselines reference
        selected_50 = select_features_fold(
            X_fold_train,
            y_fold_train,
            n_de_genes=feat_cfg["n_de_genes"],
            n_final_genes=50,
            seed=seed,
        )

        # Subset features
        X_tr_8 = X_fold_train[selected_genes].values
        X_va_8 = X_fold_val[selected_genes].values
        X_tr_50 = X_fold_train[selected_50].values
        X_va_50 = X_fold_val[selected_50].values

        y_tr = y_fold_train.values
        y_va = y_fold_val.values

        # --- Scale features ---
        X_tr_8s, X_va_8s, _, scaler_8 = scale_features(X_tr_8, X_va_8)
        X_tr_50s, X_va_50s, _, scaler_50 = scale_features(X_tr_50, X_va_50)

        pos_weight = compute_class_weights(y_tr)

        # --- 1. Hybrid Quantum Model ---
        logger.info("Training Hybrid Quantum Model...")
        hybrid_model = HybridQuantumModel(
            n_qubits=n_final_genes,
            n_layers=qm_cfg["n_layers"],
            diff_method=qm_cfg["diff_method"],
            device_name=qm_cfg["device"],
        )
        hybrid_model, h_hist = train_torch_model(
            hybrid_model,
            X_tr_8s, y_tr, X_va_8s, y_va,
            lr=qm_cfg["learning_rate"],
            batch_size=qm_cfg["batch_size"],
            max_epochs=max_epochs,
            patience=qm_cfg["early_stopping_patience"],
            pos_weight=pos_weight,
            seed=seed,
        )
        h_probs = get_torch_predictions(hybrid_model, X_va_8s)
        h_threshold = find_optimal_threshold(y_va, h_probs)
        h_metrics = compute_metrics(y_va, h_probs, h_threshold)
        h_metrics["train_time"] = h_hist["train_time"]
        h_metrics["threshold"] = h_threshold
        all_fold_results["hybrid"].append(h_metrics)

        # --- 2. Ablation: Classical replacement ---
        logger.info("Training Ablation (Classical Replacement)...")
        q_params = hybrid_model.count_parameters()
        abl_model = ClassicalAblationModel(
            n_inputs=n_final_genes,
            n_quantum_params=q_params["quantum"],
        )
        abl_model, a_hist = train_torch_model(
            abl_model,
            X_tr_8s, y_tr, X_va_8s, y_va,
            lr=qm_cfg["learning_rate"],
            batch_size=qm_cfg["batch_size"],
            max_epochs=max_epochs,
            patience=qm_cfg["early_stopping_patience"],
            pos_weight=pos_weight,
            seed=seed,
        )
        a_probs = get_torch_predictions(abl_model, X_va_8s)
        a_threshold = find_optimal_threshold(y_va, a_probs)
        a_metrics = compute_metrics(y_va, a_probs, a_threshold)
        a_metrics["train_time"] = a_hist["train_time"]
        all_fold_results["ablation"].append(a_metrics)

        # --- 3. Classical baselines on 8 genes ---
        logger.info("Training classical baselines (8 genes)...")
        svm_model, svm_info = train_svm(X_tr_8s, y_tr, X_va_8s, y_va, cl_cfg["svm"], seed)
        _, svm_probs = get_classical_predictions(svm_model, X_va_8s)
        svm_metrics = compute_metrics(y_va, svm_probs)
        svm_metrics["train_time"] = svm_info["train_time"]
        all_fold_results["svm_8"].append(svm_metrics)

        rf_model, rf_info = train_random_forest(X_tr_8s, y_tr, X_va_8s, y_va, cl_cfg["random_forest"], seed)
        _, rf_probs = get_classical_predictions(rf_model, X_va_8s)
        rf_metrics = compute_metrics(y_va, rf_probs)
        rf_metrics["train_time"] = rf_info["train_time"]
        all_fold_results["rf_8"].append(rf_metrics)

        xgb_model, xgb_info = train_xgboost(X_tr_8s, y_tr, X_va_8s, y_va, cl_cfg["xgboost"], seed)
        _, xgb_probs = get_classical_predictions(xgb_model, X_va_8s)
        xgb_metrics = compute_metrics(y_va, xgb_probs)
        xgb_metrics["train_time"] = xgb_info["train_time"]
        all_fold_results["xgb_8"].append(xgb_metrics)

        # --- 4. Classical baselines on 50 genes ---
        logger.info("Training classical baselines (50 genes)...")
        svm50, si50 = train_svm(X_tr_50s, y_tr, X_va_50s, y_va, cl_cfg["svm"], seed)
        _, sp50 = get_classical_predictions(svm50, X_va_50s)
        sm50 = compute_metrics(y_va, sp50)
        sm50["train_time"] = si50["train_time"]
        all_fold_results["svm_50"].append(sm50)

        rf50, ri50 = train_random_forest(X_tr_50s, y_tr, X_va_50s, y_va, cl_cfg["random_forest"], seed)
        _, rp50 = get_classical_predictions(rf50, X_va_50s)
        rm50 = compute_metrics(y_va, rp50)
        rm50["train_time"] = ri50["train_time"]
        all_fold_results["rf_50"].append(rm50)

        xgb50, xi50 = train_xgboost(X_tr_50s, y_tr, X_va_50s, y_va, cl_cfg["xgboost"], seed)
        _, xp50 = get_classical_predictions(xgb50, X_va_50s)
        xm50 = compute_metrics(y_va, xp50)
        xm50["train_time"] = xi50["train_time"]
        all_fold_results["xgb_50"].append(xm50)

        logger.info(
            f"Fold {fold_idx+1} results: "
            f"Hybrid AUC={h_metrics['auc']:.4f}, "
            f"Ablation AUC={a_metrics['auc']:.4f}, "
            f"SVM AUC={svm_metrics['auc']:.4f}, "
            f"RF AUC={rf_metrics['auc']:.4f}, "
            f"XGB AUC={xgb_metrics['auc']:.4f}"
        )

    return {
        "fold_results": all_fold_results,
        "selected_genes": all_selected_genes,
        "param_counts": {
            "hybrid": hybrid_model.count_parameters(),
            "ablation": abl_model.count_parameters(),
        },
    }


def run_robustness_learning_curve(
    X_trainval: pd.DataFrame,
    y_trainval: pd.Series,
    pid_trainval: pd.Series,
    config: dict,
    selected_genes: List[str],
    fractions: List[float],
    seed: int = 42,
    fast: bool = False,
) -> Dict[str, List[Dict]]:
    """Run learning curve experiment with different training sizes.

    Args:
        X_trainval: Expression data.
        y_trainval: Labels.
        pid_trainval: Patient IDs.
        config: Config dict.
        selected_genes: Pre-selected gene list.
        fractions: List of training data fractions.
        seed: Random seed.
        fast: Fast mode flag.

    Returns:
        {model_name: [metrics_per_fraction]}
    """
    set_all_seeds(seed)
    qm_cfg = config["quantum_model"]
    cl_cfg = config["classical_baselines"]
    max_epochs = config["training"]["fast_max_epochs"] if fast else qm_cfg["max_epochs"]

    # Use a single 80/20 split for learning curve
    n_genes = len(selected_genes)
    folds = get_patient_cv_folds(X_trainval, y_trainval, pid_trainval, 5, seed)
    train_idx, val_idx = folds[0]

    X_full = X_trainval[selected_genes].values
    y_full = y_trainval.values

    X_val = X_full[val_idx]
    y_val = y_full[val_idx]

    results = {"Hybrid": [], "SVM": [], "RF": [], "XGBoost": []}

    for frac in fractions:
        if frac < 1.0:
            y_fold_train = y_full[train_idx]
            from sklearn.model_selection import StratifiedShuffleSplit
            n_samples = max(8, int(len(train_idx) * frac))
            try:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
                sub_positions = next(sss.split(train_idx, y_fold_train))[0]
                sub_idx = train_idx[sub_positions]
            except Exception:
                sub_idx = train_idx[:n_samples]
        else:
            sub_idx = train_idx

        X_tr = X_full[sub_idx]
        y_tr = y_full[sub_idx]

        X_tr_s, X_va_s, _, _ = scale_features(X_tr, X_val)
        pos_weight = compute_class_weights(y_tr)

        # Hybrid
        hybrid = HybridQuantumModel(n_qubits=n_genes, n_layers=qm_cfg["n_layers"])
        hybrid, _ = train_torch_model(
            hybrid, X_tr_s, y_tr, X_va_s, y_val,
            lr=qm_cfg["learning_rate"], batch_size=qm_cfg["batch_size"],
            max_epochs=max_epochs, pos_weight=pos_weight, seed=seed,
        )
        h_probs = get_torch_predictions(hybrid, X_va_s)
        results["Hybrid"].append(compute_metrics(y_val, h_probs))

        # Classical
        svm_m, _ = train_svm(X_tr_s, y_tr, X_va_s, y_val, cl_cfg["svm"], seed)
        _, sp = get_classical_predictions(svm_m, X_va_s)
        results["SVM"].append(compute_metrics(y_val, sp))

        rf_m, _ = train_random_forest(X_tr_s, y_tr, X_va_s, y_val, cl_cfg["random_forest"], seed)
        _, rp = get_classical_predictions(rf_m, X_va_s)
        results["RF"].append(compute_metrics(y_val, rp))

        xgb_m, _ = train_xgboost(X_tr_s, y_tr, X_va_s, y_val, cl_cfg["xgboost"], seed)
        _, xp = get_classical_predictions(xgb_m, X_va_s)
        results["XGBoost"].append(compute_metrics(y_val, xp))

    return results


def run_noisy_simulation(
    X_trainval: pd.DataFrame,
    y_trainval: pd.Series,
    pid_trainval: pd.Series,
    config: dict,
    selected_genes: List[str],
    noise_level: float,
    seed: int = 42,
    fast: bool = False,
) -> Dict[str, float]:
    """Run hybrid model with depolarizing noise.

    Args:
        X_trainval: Expression data.
        y_trainval: Labels.
        pid_trainval: Patient IDs.
        config: Config dict.
        selected_genes: Gene list.
        noise_level: Depolarizing noise probability.
        seed: Random seed.
        fast: Fast mode.

    Returns:
        Metrics dict.
    """
    set_all_seeds(seed)
    qm_cfg = config["quantum_model"]
    max_epochs = config["training"]["fast_max_epochs"] if fast else qm_cfg["max_epochs"]
    n_genes = len(selected_genes)

    folds = get_patient_cv_folds(X_trainval, y_trainval, pid_trainval, 5, seed)
    train_idx, val_idx = folds[0]

    X_full = X_trainval[selected_genes].values
    y_full = y_trainval.values

    X_tr = X_full[train_idx]
    y_tr = y_full[train_idx]
    X_va = X_full[val_idx]
    y_va = y_full[val_idx]

    X_tr_s, X_va_s, _, _ = scale_features(X_tr, X_va)
    pos_weight = compute_class_weights(y_tr)

    noisy_model = HybridQuantumModel(
        n_qubits=n_genes,
        n_layers=qm_cfg["n_layers"],
        diff_method="backprop",
        device_name="default.mixed",
        noise_level=noise_level,
    )
    noisy_model, _ = train_torch_model(
        noisy_model, X_tr_s, y_tr, X_va_s, y_va,
        lr=qm_cfg["learning_rate"], batch_size=qm_cfg["batch_size"],
        max_epochs=max_epochs, pos_weight=pos_weight, seed=seed,
    )
    probs = get_torch_predictions(noisy_model, X_va_s)
    return compute_metrics(y_va, probs)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hybrid Quantum-Classical ML for Breast Cancer Detection"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic data for smoke test"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Fast mode: fewer epochs/seeds"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    fast = args.fast or config["training"].get("fast_mode", False)

    os.makedirs("results", exist_ok=True)

    logger.info("=" * 70)
    logger.info("Hybrid Quantum-Classical ML for Early Breast Cancer Detection")
    logger.info("=" * 70)
    logger.info(f"Config: {args.config}")
    logger.info(f"Synthetic: {args.synthetic}")
    logger.info(f"Fast mode: {fast}")

    # ================================================================
    # PHASE 1: Load Data
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 1: Loading Data")
    logger.info("=" * 50)

    expr_df, labels, patient_ids = load_data(config, synthetic=args.synthetic)
    logger.info(f"Data shape: {expr_df.shape}")
    logger.info(f"Label distribution: {labels.value_counts().to_dict()}")

    # ================================================================
    # PHASE 2: Preprocessing
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: Preprocessing")
    logger.info("=" * 50)

    feat_cfg = config["features"]

    # Remove low-quality genes
    expr_df = remove_low_quality_genes(
        expr_df,
        variance_threshold=feat_cfg["variance_threshold"],
        min_expression=feat_cfg["min_expression"],
    )

    # Map Ensembl to gene symbols (skip for synthetic)
    if not args.synthetic:
        try:
            probe_map = load_probemap(config["data"]["data_dir"], config["data"]["probemap_url"])
            expr_df = map_ensembl_to_symbols(expr_df, probe_map)
        except Exception as e:
            logger.warning(f"Could not load probe map: {e}. Keeping Ensembl IDs.")

    # Patient-level split
    seed = config["data"]["random_seed"]
    X_trainval, X_test, y_trainval, y_test, pid_trainval, pid_test = patient_level_split(
        expr_df, labels, patient_ids,
        test_size=config["data"]["test_size"],
        seed=seed,
    )

    logger.info(f"Train+val: {X_trainval.shape}, Test (held-out): {X_test.shape}")

    # ================================================================
    # PHASE 3: Cross-Validation Experiments
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3: Cross-Validation Experiments")
    logger.info("=" * 50)

    eval_cfg = config["evaluation"]
    n_folds = config["training"]["fast_n_folds"] if fast else eval_cfg["n_folds"]
    seeds = eval_cfg["seeds"]
    if fast:
        seeds = seeds[: config["training"]["fast_n_seeds"]]

    n_final_genes = feat_cfg["n_final_genes"]

    all_seed_results = {}
    for model_key in [
        "hybrid", "ablation", "svm_8", "rf_8", "xgb_8", "svm_50", "rf_50", "xgb_50"
    ]:
        all_seed_results[model_key] = []

    last_cv_result = None
    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\n{'#' * 60}")
        logger.info(f"SEED {seed_idx + 1}/{len(seeds)}: seed={seed}")
        logger.info(f"{'#' * 60}")

        cv_result = run_cv_experiment(
            X_trainval, y_trainval, pid_trainval, config,
            n_folds=n_folds, seed=seed, fast=fast, n_final_genes=n_final_genes,
        )
        last_cv_result = cv_result

        for model_key in all_seed_results:
            all_seed_results[model_key].extend(cv_result["fold_results"][model_key])

    # ================================================================
    # PHASE 4: Aggregate Results
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 4: Aggregating Results")
    logger.info("=" * 50)

    model_names = {
        "hybrid": "Hybrid Quantum (8 genes)",
        "ablation": "Classical Ablation (8 genes)",
        "svm_8": "SVM-RBF (8 genes)",
        "rf_8": "Random Forest (8 genes)",
        "xgb_8": "XGBoost (8 genes)",
        "svm_50": "SVM-RBF (50 genes)",
        "rf_50": "Random Forest (50 genes)",
        "xgb_50": "XGBoost (50 genes)",
    }

    aggregated = {}
    for key, name in model_names.items():
        aggregated[name] = aggregate_cv_results(all_seed_results[key])

    results_table = format_results_table(aggregated)
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS (mean ± std across all folds and seeds)")
    print("=" * 80)
    print(results_table.to_string())
    print()

    results_table.to_csv("results/cv_results.csv")
    logger.info("Saved CV results to results/cv_results.csv")

    # Parameter counts
    if last_cv_result and "param_counts" in last_cv_result:
        pc = last_cv_result["param_counts"]
        print("\nParameter Counts:")
        print(f"  Hybrid Quantum Model: {pc['hybrid']}")
        print(f"  Classical Ablation:   {pc['ablation']}")

        param_df = pd.DataFrame([
            {"Model": "Hybrid Quantum", **pc["hybrid"]},
            {"Model": "Classical Ablation", **pc["ablation"]},
        ])
        param_df.to_csv("results/parameter_counts.csv", index=False)

    # ================================================================
    # PHASE 5: Statistical Tests
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 5: Statistical Tests")
    logger.info("=" * 50)

    hybrid_aucs = [m["auc"] for m in all_seed_results["hybrid"]]
    stat_results = {}
    for key in ["ablation", "svm_8", "rf_8", "xgb_8"]:
        baseline_aucs = [m["auc"] for m in all_seed_results[key]]
        stat_results[model_names[key]] = paired_wilcoxon_test(
            hybrid_aucs, baseline_aucs, metric_name=f"AUC: Hybrid vs {model_names[key]}"
        )

    stat_df = pd.DataFrame(stat_results).T
    stat_df.to_csv("results/statistical_tests.csv")
    print("\nWilcoxon Tests (Hybrid vs Baselines):")
    print(stat_df.to_string())

    # ================================================================
    # PHASE 6: Selected Genes
    # ================================================================
    if last_cv_result and "selected_genes" in last_cv_result:
        selected = last_cv_result["selected_genes"][-1]
        with open("results/selected_genes.json", "w") as f:
            json.dump(selected, f, indent=2)
        logger.info(f"Selected genes (last fold): {selected}")

    # ================================================================
    # PHASE 7: Robustness - Learning Curve
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 7: Robustness Experiments")
    logger.info("=" * 50)

    rob_cfg = config["robustness"]
    if last_cv_result and "selected_genes" in last_cv_result:
        sel_genes = last_cv_result["selected_genes"][-1]

        logger.info("Running learning curve experiment...")
        lc_results = run_robustness_learning_curve(
            X_trainval, y_trainval, pid_trainval, config,
            sel_genes, rob_cfg["learning_curve_fractions"],
            seed=42, fast=fast,
        )
        plot_learning_curve(
            lc_results, rob_cfg["learning_curve_fractions"],
            "results/learning_curve.png"
        )

        # Save learning curve data
        lc_data = []
        for frac_idx, frac in enumerate(rob_cfg["learning_curve_fractions"]):
            for model_name, metrics_list in lc_results.items():
                row = {"fraction": frac, "model": model_name}
                row.update(metrics_list[frac_idx])
                lc_data.append(row)
        pd.DataFrame(lc_data).to_csv("results/learning_curve.csv", index=False)

        # Noisy simulation
        logger.info("Running noisy simulation experiments...")
        noise_results = []
        for nl in rob_cfg["noise_levels"]:
            logger.info(f"  Noise level: {nl}")
            try:
                nm = run_noisy_simulation(
                    X_trainval, y_trainval, pid_trainval, config,
                    sel_genes, nl, seed=42, fast=fast,
                )
                nm["noise_level"] = nl
                noise_results.append(nm)
                logger.info(f"  Noisy (p={nl}) AUC: {nm['auc']:.4f}")
            except Exception as e:
                logger.warning(f"  Noisy simulation failed at p={nl}: {e}")
                noise_results.append({"noise_level": nl, "auc": float("nan"), "error": str(e)})

        if noise_results:
            pd.DataFrame(noise_results).to_csv("results/noisy_simulation.csv", index=False)

    # ================================================================
    # PHASE 8: Explainability
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 8: Explainability")
    logger.info("=" * 50)

    if last_cv_result and "selected_genes" in last_cv_result:
        sel_genes = last_cv_result["selected_genes"][-1]

        # Train one final model for explainability
        folds = get_patient_cv_folds(X_trainval, y_trainval, pid_trainval, n_folds, seed=42)
        train_idx, val_idx = folds[0]

        X_sel = X_trainval[sel_genes].values
        y_arr = y_trainval.values

        X_tr = X_sel[train_idx]
        y_tr = y_arr[train_idx]
        X_va = X_sel[val_idx]
        y_va = y_arr[val_idx]

        X_tr_s, X_va_s, _, scaler_final = scale_features(X_tr, X_va)
        pos_weight = compute_class_weights(y_tr)

        qm_cfg = config["quantum_model"]
        max_epochs_e = config["training"]["fast_max_epochs"] if fast else qm_cfg["max_epochs"]

        # Train models for explainability
        hybrid_final = HybridQuantumModel(n_qubits=len(sel_genes), n_layers=qm_cfg["n_layers"])
        hybrid_final, _ = train_torch_model(
            hybrid_final, X_tr_s, y_tr, X_va_s, y_va,
            lr=qm_cfg["learning_rate"], batch_size=qm_cfg["batch_size"],
            max_epochs=max_epochs_e, pos_weight=pos_weight, seed=42,
        )

        cl_cfg = config["classical_baselines"]
        svm_final, _ = train_svm(X_tr_s, y_tr, X_va_s, y_va, cl_cfg["svm"], seed=42)
        rf_final, _ = train_random_forest(X_tr_s, y_tr, X_va_s, y_va, cl_cfg["random_forest"], seed=42)
        xgb_final, _ = train_xgboost(X_tr_s, y_tr, X_va_s, y_va, cl_cfg["xgboost"], seed=42)

        # Permutation importance
        logger.info("Computing permutation importance...")
        models_for_perm = {
            "Hybrid": (hybrid_final, True),
            "SVM": (svm_final, False),
            "RF": (rf_final, False),
            "XGBoost": (xgb_final, False),
        }
        all_importances = {}
        for mname, (model, is_torch) in models_for_perm.items():
            imp = permutation_importance_manual(
                model, X_va_s, y_va, sel_genes,
                n_repeats=config["explainability"]["n_permutations"],
                seed=42, is_torch=is_torch,
            )
            all_importances[mname] = imp
            plot_permutation_importance(imp, mname, f"results/importance_{mname.lower()}.png")
            imp.to_csv(f"results/importance_{mname.lower()}.csv", index=False)

        # SHAP for hybrid
        logger.info("Computing SHAP values for hybrid model...")
        shap_vals = shap_explain_hybrid(
            hybrid_final, X_tr_s, X_va_s, sel_genes,
            n_samples=config["explainability"]["shap_nsamples"],
        )
        if shap_vals is not None:
            plot_shap_summary(shap_vals, X_va_s, sel_genes, "results/shap_summary.png")

        # Pathway enrichment
        logger.info("Running pathway enrichment...")
        # Filter out synthetic gene names for enrichment
        real_genes = [g for g in sel_genes if not g.startswith("ENSG")]
        if real_genes:
            enrich_results = pathway_enrichment(
                real_genes,
                config["explainability"]["enrichr_gene_sets"],
                output_dir="results",
            )
            if enrich_results is not None:
                enrich_results.to_csv("results/pathway_enrichment.csv", index=False)

        # Save model for Streamlit
        torch.save(hybrid_final.state_dict(), "results/trained_model.pt")
        # Save scaler
        import pickle
        with open("results/scaler.pkl", "wb") as f:
            pickle.dump(scaler_final, f)
        logger.info("Saved trained model and scaler.")

    # ================================================================
    # PHASE 9: Held-out Test Set Evaluation (ONCE)
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 9: Held-Out Test Set Evaluation")
    logger.info("=" * 50)

    if last_cv_result and "selected_genes" in last_cv_result:
        sel_genes = last_cv_result["selected_genes"][-1]

        # Retrain on FULL train+val using last selected genes
        X_train_full = X_trainval[sel_genes].values
        y_train_full = y_trainval.values
        X_test_sel = X_test[sel_genes].values
        y_test_arr = y_test.values

        # Split a small portion for validation (for early stopping / threshold)
        from sklearn.model_selection import train_test_split
        tr_idx, va_idx = train_test_split(
            np.arange(len(y_train_full)),
            test_size=0.15, stratify=y_train_full, random_state=42
        )

        X_tr_raw = X_train_full[tr_idx]
        y_tr_final = y_train_full[tr_idx]
        X_va_raw = X_train_full[va_idx]
        y_va_final = y_train_full[va_idx]

        # Scale: fit on training portion only!
        X_tr_final, X_va_final, X_te_s, scaler_test = scale_features(
            X_tr_raw, X_va_raw, X_test_sel
        )

        pos_weight = compute_class_weights(y_tr_final)
        qm_cfg = config["quantum_model"]
        max_epochs_t = config["training"]["fast_max_epochs"] if fast else qm_cfg["max_epochs"]

        # Train final hybrid
        hybrid_test = HybridQuantumModel(n_qubits=len(sel_genes), n_layers=qm_cfg["n_layers"])
        hybrid_test, _ = train_torch_model(
            hybrid_test, X_tr_final, y_tr_final, X_va_final, y_va_final,
            lr=qm_cfg["learning_rate"], batch_size=qm_cfg["batch_size"],
            max_epochs=max_epochs_t, pos_weight=pos_weight, seed=42,
        )

        # Choose threshold on validation, apply to test
        h_val_probs = get_torch_predictions(hybrid_test, X_va_final)
        h_test_threshold = find_optimal_threshold(y_va_final, h_val_probs)

        h_test_probs = get_torch_predictions(hybrid_test, X_te_s)
        h_test_metrics = compute_metrics(y_test_arr, h_test_probs, h_test_threshold)

        # Classical baselines on test
        cl_cfg = config["classical_baselines"]
        svm_test, _ = train_svm(X_tr_final, y_tr_final, X_va_final, y_va_final, cl_cfg["svm"], 42)
        _, svm_tp = get_classical_predictions(svm_test, X_te_s)
        svm_test_m = compute_metrics(y_test_arr, svm_tp)

        rf_test, _ = train_random_forest(X_tr_final, y_tr_final, X_va_final, y_va_final, cl_cfg["random_forest"], 42)
        _, rf_tp = get_classical_predictions(rf_test, X_te_s)
        rf_test_m = compute_metrics(y_test_arr, rf_tp)

        xgb_test, _ = train_xgboost(X_tr_final, y_tr_final, X_va_final, y_va_final, cl_cfg["xgboost"], 42)
        _, xgb_tp = get_classical_predictions(xgb_test, X_te_s)
        xgb_test_m = compute_metrics(y_test_arr, xgb_tp)

        # Ablation on test
        abl_test = ClassicalAblationModel(
            n_inputs=len(sel_genes),
            n_quantum_params=hybrid_test.count_parameters()["quantum"],
        )
        abl_test, _ = train_torch_model(
            abl_test, X_tr_final, y_tr_final, X_va_final, y_va_final,
            lr=qm_cfg["learning_rate"], batch_size=qm_cfg["batch_size"],
            max_epochs=max_epochs_t, pos_weight=pos_weight, seed=42,
        )
        abl_val_probs = get_torch_predictions(abl_test, X_va_final)
        abl_threshold = find_optimal_threshold(y_va_final, abl_val_probs)
        abl_tp = get_torch_predictions(abl_test, X_te_s)
        abl_test_m = compute_metrics(y_test_arr, abl_tp, abl_threshold)

        test_results = {
            "Hybrid Quantum": h_test_metrics,
            "Classical Ablation": abl_test_m,
            "SVM-RBF": svm_test_m,
            "Random Forest": rf_test_m,
            "XGBoost": xgb_test_m,
        }

        print("\n" + "=" * 80)
        print("HELD-OUT TEST SET RESULTS")
        print("=" * 80)
        test_df = pd.DataFrame(test_results).T
        test_df.index.name = "Model"
        print(test_df.to_string())
        test_df.to_csv("results/test_results.csv")
        logger.info("Saved test results to results/test_results.csv")

        # ROC curves for test set
        roc_data = {}
        for name, probs in [
            ("Hybrid Quantum", h_test_probs),
            ("SVM-RBF", svm_tp),
            ("Random Forest", rf_tp),
            ("XGBoost", xgb_tp),
        ]:
            if len(np.unique(y_test_arr)) >= 2:
                fpr, tpr, _ = roc_curve(y_test_arr, probs)
                roc_data[name] = {
                    "fpr": fpr, "tpr": tpr,
                    "auc": roc_auc_score(y_test_arr, probs)
                }

        if roc_data:
            plot_roc_comparison(roc_data, "results/roc_comparison.png")

    # ================================================================
    # DONE
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Results saved to: results/")
    logger.info("  - cv_results.csv")
    logger.info("  - test_results.csv")
    logger.info("  - parameter_counts.csv")
    logger.info("  - statistical_tests.csv")
    logger.info("  - selected_genes.json")
    logger.info("  - learning_curve.csv / .png")
    logger.info("  - noisy_simulation.csv")
    logger.info("  - importance_*.csv / .png")
    logger.info("  - roc_comparison.png")


if __name__ == "__main__":
    main()

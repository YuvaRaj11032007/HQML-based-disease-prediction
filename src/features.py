"""
Feature selection and preprocessing for TCGA-BRCA gene expression data.

Pipeline:
1. Remove near-zero variance and low-expression genes.
2. Map Ensembl IDs to gene symbols (optional).
3. Patient-level stratified train/test split.
4. Within each CV fold: differential expression (t-test) -> mutual information ranking.
5. MinMaxScaler to [0, pi] fit on training portion only.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def remove_low_quality_genes(
    expr_df: pd.DataFrame,
    variance_threshold: float = 0.1,
    min_expression: float = 1.0,
) -> pd.DataFrame:
    """Remove genes with near-zero variance or very low expression.

    Args:
        expr_df: Samples x genes expression DataFrame.
        variance_threshold: Minimum variance to keep a gene.
        min_expression: Minimum mean expression to keep a gene.

    Returns:
        Filtered DataFrame.
    """
    n_before = expr_df.shape[1]

    # Remove low variance
    variances = expr_df.var(axis=0)
    mask_var = variances > variance_threshold

    # Remove low expression
    means = expr_df.mean(axis=0)
    mask_expr = means > min_expression

    mask = mask_var & mask_expr
    expr_df = expr_df.loc[:, mask]

    logger.info(
        f"Gene filtering: {n_before} -> {expr_df.shape[1]} genes "
        f"(removed {n_before - expr_df.shape[1]} low-quality genes)"
    )
    return expr_df


def map_ensembl_to_symbols(
    expr_df: pd.DataFrame, probe_map: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Map Ensembl IDs to gene symbols using probe map.

    Strips version suffix from Ensembl IDs (e.g., ENSG00000141510.16 -> ENSG00000141510).
    If multiple Ensembl IDs map to the same gene, keeps the one with highest variance.

    Args:
        expr_df: Expression DataFrame with Ensembl IDs as columns.
        probe_map: DataFrame with 'id' and 'gene' columns.

    Returns:
        DataFrame with gene symbols as columns.
    """
    if probe_map is None:
        logger.info("No probe map provided; keeping Ensembl IDs.")
        return expr_df

    # Strip version from Ensembl IDs
    stripped = {col: col.split(".")[0] for col in expr_df.columns}
    expr_renamed = expr_df.rename(columns=stripped)

    # Build mapping
    pm = probe_map.copy()
    pm["id_stripped"] = pm["id"].str.split(".").str[0]
    id_to_gene = dict(zip(pm["id_stripped"], pm["gene"]))

    # Map columns
    new_cols = []
    for col in expr_renamed.columns:
        new_cols.append(id_to_gene.get(col, col))
    expr_renamed.columns = new_cols

    # Handle duplicates: keep highest variance
    if expr_renamed.columns.duplicated().any():
        # Group by column name, keep the column with max variance
        variances = expr_renamed.var(axis=0)
        keep_cols = []
        seen = set()
        for col in expr_renamed.columns:
            if col not in seen:
                # Find all indices with this column name
                dup_mask = expr_renamed.columns == col
                if dup_mask.sum() > 1:
                    dup_vars = variances[dup_mask]
                    best_idx = dup_vars.values.argmax()
                    keep_cols.append(expr_renamed.columns[dup_mask].tolist()[best_idx])
                else:
                    keep_cols.append(col)
                seen.add(col)

        # Deduplicate by selecting unique columns
        expr_renamed = expr_renamed.loc[:, ~expr_renamed.columns.duplicated(keep="first")]
        n_dups = len(new_cols) - expr_renamed.shape[1]
        if n_dups > 0:
            logger.info(f"Removed {n_dups} duplicate gene symbol mappings (kept highest variance).")

    logger.info(f"Mapped to gene symbols: {expr_renamed.shape[1]} genes")
    return expr_renamed


def patient_level_split(
    expr_df: pd.DataFrame,
    labels: pd.Series,
    patient_ids: pd.Series,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split data by patient so same patient never appears in both train and test.

    Uses stratified splitting at the patient level to maintain class balance.

    Args:
        expr_df: Expression data (samples x genes).
        labels: Sample labels (0 or 1).
        patient_ids: Patient IDs per sample.
        test_size: Fraction for held-out test.
        seed: Random seed.

    Returns:
        (X_train, X_test, y_train, y_test, pid_train, pid_test)
    """
    rng = np.random.RandomState(seed)

    # Get unique patients and their labels
    # If a patient has multiple samples, use majority label
    patient_label = patient_ids.to_frame().join(labels)
    patient_label_map = patient_label.groupby("patient_id")["label"].agg(
        lambda x: x.mode().iloc[0]
    )
    unique_patients = patient_label_map.index.values
    unique_labels = patient_label_map.values

    # Stratified split at patient level
    n_test = max(1, int(len(unique_patients) * test_size))
    indices = np.arange(len(unique_patients))
    rng.shuffle(indices)

    # Simple stratified split
    from sklearn.model_selection import train_test_split

    train_patients, test_patients = train_test_split(
        unique_patients,
        test_size=test_size,
        stratify=unique_labels,
        random_state=seed,
    )

    train_mask = patient_ids.isin(train_patients)
    test_mask = patient_ids.isin(test_patients)

    X_train = expr_df.loc[train_mask]
    X_test = expr_df.loc[test_mask]
    y_train = labels.loc[train_mask]
    y_test = labels.loc[test_mask]
    pid_train = patient_ids.loc[train_mask]
    pid_test = patient_ids.loc[test_mask]

    # Verify no patient overlap
    train_set = set(pid_train.unique())
    test_set = set(pid_test.unique())
    assert train_set.isdisjoint(test_set), "Patient overlap between train and test!"

    logger.info(
        f"Patient-level split: {len(train_set)} train patients ({len(X_train)} samples), "
        f"{len(test_set)} test patients ({len(X_test)} samples)"
    )
    logger.info(
        f"Train class balance: {(y_train == 1).sum()} tumor, {(y_train == 0).sum()} normal"
    )
    logger.info(
        f"Test class balance: {(y_test == 1).sum()} tumor, {(y_test == 0).sum()} normal"
    )

    return X_train, X_test, y_train, y_test, pid_train, pid_test


def select_features_fold(
    X_train_fold: pd.DataFrame,
    y_train_fold: pd.Series,
    n_de_genes: int = 200,
    n_final_genes: int = 8,
    seed: int = 42,
) -> List[str]:
    """Select features within a single CV fold (using training portion only).

    Pipeline:
    1. t-test for differential expression, keep top n_de_genes.
    2. Mutual information ranking, keep top n_final_genes.

    Args:
        X_train_fold: Training expression data for this fold.
        y_train_fold: Training labels for this fold.
        n_de_genes: Number of top DE genes from t-test.
        n_final_genes: Final number of genes to select.
        seed: Random seed for MI estimation.

    Returns:
        List of selected gene names.
    """
    # Step 1: t-test for differential expression
    tumor_mask = y_train_fold == 1
    normal_mask = y_train_fold == 0

    tumor_data = X_train_fold.loc[tumor_mask]
    normal_data = X_train_fold.loc[normal_mask]

    t_stats = []
    p_vals = []
    for gene in X_train_fold.columns:
        t_stat, p_val = stats.ttest_ind(
            tumor_data[gene].values,
            normal_data[gene].values,
            equal_var=False,  # Welch's t-test
        )
        t_stats.append(abs(t_stat))
        p_vals.append(p_val)

    de_results = pd.DataFrame(
        {"gene": X_train_fold.columns, "t_stat": t_stats, "p_value": p_vals}
    )
    de_results = de_results.sort_values("t_stat", ascending=False)

    # Keep top n_de_genes
    top_de = de_results.head(n_de_genes)["gene"].tolist()
    X_de = X_train_fold[top_de]

    logger.debug(f"Top DE gene: {top_de[0]} (t={de_results.iloc[0]['t_stat']:.2f})")

    # Step 2: Mutual information ranking
    mi_scores = mutual_info_classif(
        X_de.values, y_train_fold.values, random_state=seed, n_neighbors=5
    )
    mi_df = pd.DataFrame({"gene": top_de, "mi_score": mi_scores})
    mi_df = mi_df.sort_values("mi_score", ascending=False)

    selected = mi_df.head(n_final_genes)["gene"].tolist()
    logger.info(
        f"Selected {n_final_genes} genes: {selected[:5]}{'...' if len(selected) > 5 else ''}"
    )

    return selected


def select_features_hierarchical(
    X_train_fold: pd.DataFrame,
    y_train_fold: pd.Series,
    n_de_genes: int = 200,
    n_qubit_genes: int = 8,
    n_reference_genes: int = 50,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Select features with a single shared Welch's t-test step.

    Pipeline:
    1. Single Welch's t-test step on training partition -> top n_de_genes (top 200).
    2. Mutual information ranking on the top DE genes -> top n_qubit_genes (8)
       and top n_reference_genes (50).

    Args:
        X_train_fold: Training expression data for this fold.
        y_train_fold: Training labels for this fold.
        n_de_genes: Top DE genes from t-test (default 200).
        n_qubit_genes: Final qubit features (default 8).
        n_reference_genes: Reference baseline features (default 50).
        seed: Random seed for MI estimation.

    Returns:
        Tuple of (selected_8_genes, selected_50_genes, top_200_de_genes).
    """
    tumor_mask = y_train_fold == 1
    normal_mask = y_train_fold == 0

    tumor_data = X_train_fold.loc[tumor_mask]
    normal_data = X_train_fold.loc[normal_mask]

    t_stats = []
    p_vals = []
    for gene in X_train_fold.columns:
        t_stat, p_val = stats.ttest_ind(
            tumor_data[gene].values,
            normal_data[gene].values,
            equal_var=False,  # Welch's t-test
        )
        t_stats.append(abs(t_stat))
        p_vals.append(p_val)

    de_results = pd.DataFrame(
        {"gene": X_train_fold.columns, "t_stat": t_stats, "p_value": p_vals}
    )
    de_results = de_results.sort_values("t_stat", ascending=False)

    top_de = de_results.head(n_de_genes)["gene"].tolist()
    X_de = X_train_fold[top_de]

    # Mutual information ranking once on the top DE genes
    mi_scores = mutual_info_classif(
        X_de.values, y_train_fold.values, random_state=seed, n_neighbors=5
    )
    mi_df = pd.DataFrame({"gene": top_de, "mi_score": mi_scores})
    mi_df = mi_df.sort_values("mi_score", ascending=False)

    selected_qubit = mi_df.head(n_qubit_genes)["gene"].tolist()
    selected_reference = mi_df.head(n_reference_genes)["gene"].tolist()

    logger.info(
        f"Hierarchical feature selection: top {len(top_de)} DE -> "
        f"{len(selected_qubit)} qubit genes, {len(selected_reference)} reference genes"
    )

    return selected_qubit, selected_reference, top_de


def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    feature_range: Tuple[float, float] = (0.0, np.pi),
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], MinMaxScaler]:
    """Scale features to [0, pi] using MinMaxScaler fit on train only.

    Args:
        X_train: Training features.
        X_val: Validation features.
        X_test: Optional test features.
        feature_range: Target range (default [0, pi]).

    Returns:
        (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def get_patient_cv_folds(
    X: pd.DataFrame,
    y: pd.Series,
    patient_ids: pd.Series,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate patient-level stratified CV folds.

    Ensures no patient appears in both train and validation within a fold.

    Args:
        X: Feature matrix.
        y: Labels.
        patient_ids: Patient IDs.
        n_folds: Number of folds.
        seed: Random seed.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    # Use StratifiedGroupKFold to respect patient grouping
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for train_idx, val_idx in sgkf.split(X, y, groups=patient_ids):
        # Verify no patient overlap
        train_pids = set(patient_ids.iloc[train_idx].unique())
        val_pids = set(patient_ids.iloc[val_idx].unique())
        assert train_pids.isdisjoint(val_pids), "Patient overlap in CV fold!"
        folds.append((train_idx, val_idx))

    return folds

"""
Data loading, downloading, and synthetic data generation for TCGA-BRCA.

Downloads expression matrix and phenotype data from UCSC Xena (GDC hub).
Extracts labels from TCGA barcode sample-type codes:
  - 01 = primary tumor (label 1)
  - 11 = solid tissue normal (label 0)
"""

import gzip
import io
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_file(url: str, dest: str, chunk_size: int = 8192) -> None:
    """Download a file from URL with progress bar.

    Args:
        url: Source URL.
        dest: Destination file path.
        chunk_size: Download chunk size in bytes.

    Raises:
        RuntimeError: If download fails.
    """
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        logger.info(f"File already exists: {dest}")
        return

    logger.info(f"Downloading {url} -> {dest}")
    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Failed to download {url}.\n"
            f"Error: {e}\n"
            f"Please check your internet connection or download manually from:\n"
            f"  {url}\n"
            f"and place the file at: {dest}"
        ) from e

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=Path(dest).name
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info(f"Downloaded: {dest}")


def load_probemap(data_dir: str, url: str) -> pd.DataFrame:
    """Download and load the Ensembl-to-gene-symbol probe map.

    Args:
        data_dir: Directory for data files.
        url: URL of the probeMap file.

    Returns:
        DataFrame with 'id' (Ensembl) and 'gene' (symbol) columns.
    """
    dest = os.path.join(data_dir, "probeMap.tsv")
    download_file(url, dest)
    probe_map = pd.read_csv(dest, sep="\t", usecols=["id", "gene"])
    logger.info(f"Loaded probeMap: {probe_map.shape[0]} entries")
    return probe_map


def extract_sample_type(barcode: str) -> Optional[str]:
    """Extract sample type code from TCGA barcode (characters 14-15, 0-indexed 13:15).

    Args:
        barcode: Full TCGA barcode string.

    Returns:
        Two-character sample type code, or None if barcode is too short.
    """
    if len(barcode) >= 15:
        return barcode[13:15]
    return None


def extract_patient_id(barcode: str) -> str:
    """Extract patient ID from TCGA barcode (first 12 characters).

    Args:
        barcode: Full TCGA barcode string.

    Returns:
        12-character patient ID.
    """
    return barcode[:12]


def load_tcga_data(
    config: dict,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Download and load TCGA-BRCA expression data with labels.

    Args:
        config: Configuration dictionary with data URLs and paths.

    Returns:
        Tuple of (expression_df, labels, patient_ids) where:
          - expression_df: genes x samples DataFrame (already log2(x+1))
          - labels: Series of 0 (normal) or 1 (tumor)
          - patient_ids: Series of patient IDs
    """
    data_cfg = config["data"]
    data_dir = data_cfg["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    # Check for existing downloaded files first
    candidate_files = [
        os.path.join(data_dir, "TCGA-BRCA.htseq_fpkm-uq.tsv.gz"),
        os.path.join(data_dir, "TCGA-BRCA.HiSeqV2.tsv.gz"),
        os.path.join(data_dir, "TCGA-BRCA.htseq_fpkm-uq.tsv"),
        os.path.join(data_dir, "TCGA-BRCA.HiSeqV2.tsv"),
    ]
    expr_file = None
    for cand in candidate_files:
        if os.path.exists(cand):
            expr_file = cand
            logger.info(f"Using local TCGA expression matrix: {expr_file}")
            break

    if expr_file is None:
        expr_gz = os.path.join(data_dir, "TCGA-BRCA.htseq_fpkm-uq.tsv.gz")
        try:
            download_file(data_cfg["expression_url"], expr_gz)
            expr_file = expr_gz
        except Exception as e:
            logger.warning(
                f"Download from primary URL failed ({e}). "
                f"Falling back to verified UCSC Xena TCGA Hub RNA-seq dataset..."
            )
            fallback_url = "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz"
            fallback_gz = os.path.join(data_dir, "TCGA-BRCA.HiSeqV2.tsv.gz")
            download_file(fallback_url, fallback_gz)
            expr_file = fallback_gz

    # --- Load expression matrix (genes as rows, samples as columns) ---
    logger.info(f"Loading expression matrix from {expr_file} (this may take a minute)...")
    expr_df = pd.read_csv(expr_file, sep="\t", index_col=0)
    logger.info(f"Raw expression matrix: {expr_df.shape[0]} genes x {expr_df.shape[1]} samples")

    # --- Filter samples by barcode sample-type ---
    valid_types = set(data_cfg.get("valid_sample_types", ["01", "11"]))
    sample_info = []
    for col in expr_df.columns:
        st = extract_sample_type(col)
        if st in valid_types:
            label = 1 if st == "01" else 0
            pid = extract_patient_id(col)
            sample_info.append({"barcode": col, "label": label, "patient_id": pid})

    info_df = pd.DataFrame(sample_info)
    logger.info(
        f"Filtered to {len(info_df)} samples: "
        f"{(info_df['label'] == 1).sum()} tumor, {(info_df['label'] == 0).sum()} normal"
    )

    # Subset expression to valid samples
    expr_df = expr_df[info_df["barcode"].values]

    # Transpose: samples x genes
    expr_df = expr_df.T
    expr_df.index = info_df["barcode"].values

    labels = pd.Series(info_df["label"].values, index=info_df["barcode"].values, name="label")
    patient_ids = pd.Series(
        info_df["patient_id"].values, index=info_df["barcode"].values, name="patient_id"
    )

    return expr_df, labels, patient_ids


def generate_synthetic_data(
    n_tumor: int = 120,
    n_normal: int = 30,
    n_genes: int = 500,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate synthetic data mimicking TCGA-BRCA structure for smoke tests.

    Creates fake gene expression data with tumor/normal imbalance similar
    to the real dataset (~4:1 ratio).

    Args:
        n_tumor: Number of tumor samples.
        n_normal: Number of normal samples.
        n_genes: Number of genes (features).
        seed: Random seed.

    Returns:
        Tuple of (expression_df, labels, patient_ids) with same structure
        as load_tcga_data output.
    """
    rng = np.random.RandomState(seed)
    n_total = n_tumor + n_normal

    # Create Ensembl-like gene IDs
    gene_ids = [f"ENSG{i:011d}.{rng.randint(1,15)}" for i in range(n_genes)]

    # Generate expression: tumor has slightly different profile for ~50 genes
    base_expr = rng.uniform(0, 15, size=(n_total, n_genes))  # log2(FPKM-UQ + 1) range

    # Make first 50 genes differentially expressed
    n_de = 50
    labels_arr = np.array([1] * n_tumor + [0] * n_normal)
    for i in range(n_de):
        # Tumor has higher expression for some genes, lower for others
        direction = 1.0 if i % 2 == 0 else -1.0
        effect = rng.uniform(2.0, 5.0)
        base_expr[labels_arr == 1, i] += direction * effect
        # Add some noise
        base_expr[:, i] += rng.normal(0, 0.5, n_total)

    base_expr = np.clip(base_expr, 0, 20)

    # Create TCGA-like barcodes
    barcodes = []
    patient_ids_list = []
    for i in range(n_total):
        pid = f"TCGA-{chr(65 + i // 26)}{chr(65 + i % 26)}-{i:04d}"
        sample_type = "01" if labels_arr[i] == 1 else "11"
        barcode = f"{pid}-{sample_type}A-11D-A1B2-09"
        barcodes.append(barcode)
        patient_ids_list.append(pid)

    expr_df = pd.DataFrame(base_expr, index=barcodes, columns=gene_ids)
    labels = pd.Series(labels_arr, index=barcodes, name="label")
    patient_ids = pd.Series(patient_ids_list, index=barcodes, name="patient_id")

    logger.info(
        f"Generated synthetic data: {n_total} samples ({n_tumor} tumor, {n_normal} normal), "
        f"{n_genes} genes"
    )

    return expr_df, labels, patient_ids


def load_data(
    config: dict, synthetic: bool = False
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Main entry point: load real TCGA data or synthetic data.

    Args:
        config: Configuration dictionary.
        synthetic: If True, generate synthetic data instead of downloading.

    Returns:
        Tuple of (expression_df, labels, patient_ids).
    """
    if synthetic:
        logger.info("Using synthetic data for smoke test.")
        return generate_synthetic_data()
    else:
        return load_tcga_data(config)


def compute_imbalance_ratio(labels: pd.Series) -> Tuple[float, str]:
    """Compute exact class imbalance ratio (tumor:normal) from empirical data.

    Args:
        labels: Series of binary labels (1=tumor, 0=normal).

    Returns:
        (ratio_float, ratio_str) e.g., (10.9, "10.9:1").
    """
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    ratio = float(n_pos) / max(float(n_neg), 1.0)
    ratio_str = f"{ratio:.1f}:1"
    logger.info(f"Empirical class distribution: {n_pos} tumor, {n_neg} normal -> {ratio_str} imbalance")
    return ratio, ratio_str


def rank_normalize_cohort(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Rank-normalize expression data per sample across genes to [0, 1].

    Normalizes each cohort independently across shared genes so that
    cross-platform expression distributions (e.g., RNA-seq FPKM-UQ vs Affymetrix microarray)
    are harmonized prior to feature scaling.

    Args:
        expr_df: Samples x genes DataFrame.

    Returns:
        Normalized DataFrame with percentile ranks in [0, 1].
    """
    ranked = expr_df.rank(axis=1, pct=True)
    return ranked


def load_geo_data(
    data_dir: str = "data/geo",
    accession: str = "GSE42568",
    common_genes: Optional[list] = None,
    synthetic: bool = False,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load or simulate an external GEO breast cancer validation cohort (GSE42568).

    GSE42568 contains 104 breast tumor samples and 17 normal breast tissue controls
    profiled on the Affymetrix HG-U133 Plus 2.0 (GPL570) microarray platform.

    Args:
        data_dir: Directory for GEO data cache.
        accession: GEO accession ID (default: 'GSE42568').
        common_genes: Optional list of genes to subset to.
        synthetic: If True, generate synthetic GEO data.
        seed: Random seed.

    Returns:
        Tuple of (geo_expr_df, geo_labels).
    """
    os.makedirs(data_dir, exist_ok=True)
    matrix_gz = os.path.join(data_dir, f"{accession}_series_matrix.txt.gz")
    annot_gz = os.path.join(data_dir, "GPL570.annot.gz")

    if not synthetic:
        # Check if local files exist, or try to download
        geo_matrix_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE42nnn/{accession}/matrix/{accession}_series_matrix.txt.gz"
        geo_annot_url = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"

        try:
            if not os.path.exists(matrix_gz):
                download_file(geo_matrix_url, matrix_gz)
            if not os.path.exists(annot_gz):
                download_file(geo_annot_url, annot_gz)
        except Exception as e:
            logger.warning(f"Failed downloading GEO files: {e}. Falling back to synthetic cohort.")

        if os.path.exists(matrix_gz) and os.path.exists(annot_gz):
            logger.info(f"Parsing real GEO dataset ({accession}) and platform annotation (GPL570)...")
            try:
                # 1. Parse probe to symbol mapping from GPL570
                probe_to_symbol = {}
                with gzip.open(annot_gz, "rt", encoding="utf-8", errors="ignore") as f:
                    header_found = False
                    id_col = 0
                    sym_col = 2
                    for line in f:
                        if line.startswith("#") or not line.strip():
                            continue
                        parts = line.strip().split("\t")
                        if not header_found:
                            if "ID" in parts and any("Gene symbol" in p or "Gene Symbol" in p for p in parts):
                                header_found = True
                                id_col = parts.index("ID")
                                for p_idx, p_name in enumerate(parts):
                                    if "Gene symbol" in p_name or "Gene Symbol" in p_name:
                                        sym_col = p_idx
                                        break
                            continue
                        if len(parts) > max(id_col, sym_col):
                            probe = parts[id_col].strip()
                            sym = parts[sym_col].strip()
                            if sym and sym != "---":
                                probe_to_symbol[probe] = sym.split("///")[0].strip()

                # 2. Parse series matrix table
                sample_ids = []
                labels = []
                lines_data = []
                in_table = False

                with gzip.open(matrix_gz, "rt", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.startswith("!Sample_geo_accession"):
                            sample_ids = [x.strip('"\n ') for x in line.split("\t")[1:]]
                        elif line.startswith("!Sample_characteristics_ch1\t\"tissue:"):
                            tissues = [x.strip('"\n ') for x in line.split("\t")[1:]]
                            labels = [1 if "cancer" in t.lower() or "tumor" in t.lower() else 0 for t in tissues]
                        elif line.startswith("!series_matrix_table_begin"):
                            in_table = True
                            continue
                        elif line.startswith("!series_matrix_table_end"):
                            break
                        elif in_table:
                            lines_data.append(line)

                data_str = "".join(lines_data)
                df = pd.read_csv(io.StringIO(data_str), sep="\t", index_col=0)
                df.index = df.index.astype(str)

                # Map probe IDs to gene symbols
                df["gene_symbol"] = df.index.map(probe_to_symbol)
                df = df.dropna(subset=["gene_symbol"])
                df["mean_expr"] = df[sample_ids].mean(axis=1)
                df = df.sort_values("mean_expr", ascending=False).drop_duplicates(subset=["gene_symbol"])
                df = df.set_index("gene_symbol")[sample_ids]

                expr_df = df.T
                labels_series = pd.Series(labels, index=sample_ids, name="label")

                # If common_genes provided, subset or align
                if common_genes is not None:
                    missing = [g for g in common_genes if g not in expr_df.columns]
                    if missing:
                        logger.warning(f"GEO cohort missing {len(missing)} genes: {missing}. Imputing 0.0.")
                        for mg in missing:
                            expr_df[mg] = 0.0
                    expr_df = expr_df[common_genes]

                logger.info(
                    f"Successfully loaded real GEO cohort ({accession}): "
                    f"{expr_df.shape[0]} samples ({(labels_series==1).sum()} tumor, "
                    f"{(labels_series==0).sum()} normal), {expr_df.shape[1]} genes"
                )
                return expr_df, labels_series
            except Exception as e:
                logger.error(f"Error parsing real GEO data: {e}. Falling back to synthetic cohort.")

    # Synthetic fallback mode
    logger.info(
        f"Generating synthetic external validation cohort ({accession}: 104 tumor, 17 normal samples)..."
    )
    rng = np.random.RandomState(seed)
    n_tumor = 104
    n_normal = 17
    n_total = n_tumor + n_normal
    labels_arr = np.array([1] * n_tumor + [0] * n_normal)

    genes = common_genes if common_genes is not None else [f"GENE_{i}" for i in range(200)]
    base_expr = rng.uniform(4.0, 14.0, size=(n_total, len(genes)))
    for i in range(min(len(genes), 50)):
        direction = 1.0 if i % 2 == 0 else -1.0
        base_expr[labels_arr == 1, i] += direction * rng.uniform(1.0, 2.5)

    sample_ids = [f"GSM1044{i:03d}" for i in range(n_total)]
    geo_df = pd.DataFrame(base_expr, index=sample_ids, columns=genes)
    geo_labels = pd.Series(labels_arr, index=sample_ids, name="label")
    return geo_df, geo_labels

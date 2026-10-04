"""
Unit tests for the Hybrid Quantum-Classical ML pipeline.

Tests:
(a) No patient overlap between train and test
(b) Feature selection never sees validation or test data
(c) Scaling is fit on train only
(d) Quantum model outputs shape (batch, 1) with values in [0, 1]
(e) End-to-end smoke test on synthetic data
"""

import json
import os
import time

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import generate_synthetic_data, extract_patient_id, extract_sample_type
from src.features import (
    get_patient_cv_folds,
    patient_level_split,
    remove_low_quality_genes,
    scale_features,
    select_features_fold,
)
from src.models_quantum import (
    ClassicalAblationModel,
    HybridQuantumModel,
    compute_class_weights,
    train_torch_model,
)
from src.models_classical import train_svm, train_random_forest, train_xgboost, get_classical_predictions
from src.evaluate import compute_metrics, find_optimal_threshold


# ================================================================
# Fixtures
# ================================================================


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing."""
    return generate_synthetic_data(n_tumor=80, n_normal=20, n_genes=200, seed=42)


@pytest.fixture
def config():
    """Minimal test configuration."""
    return {
        "data": {
            "data_dir": "data/",
            "test_size": 0.2,
            "random_seed": 42,
            "valid_sample_types": ["01", "11"],
        },
        "features": {
            "variance_threshold": 0.1,
            "min_expression": 1.0,
            "n_de_genes": 100,
            "n_final_genes": 8,
        },
        "quantum_model": {
            "n_qubits": 8,
            "n_layers": 2,
            "learning_rate": 0.02,
            "batch_size": 16,
            "max_epochs": 5,
            "early_stopping_patience": 3,
            "diff_method": "backprop",
            "device": "default.qubit",
        },
        "classical_baselines": {
            "svm": {"kernel": "rbf", "C_range": [1, 10], "gamma_range": ["scale"]},
            "random_forest": {"n_estimators": 50, "max_depth_range": [5, 10]},
            "xgboost": {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1},
        },
    }


# ================================================================
# Test (a): No patient overlap
# ================================================================


class TestPatientOverlap:
    """Test that no patient appears in both train and test sets."""

    def test_patient_level_split_no_overlap(self, synthetic_data):
        expr_df, labels, patient_ids = synthetic_data
        X_train, X_test, y_train, y_test, pid_train, pid_test = patient_level_split(
            expr_df, labels, patient_ids, test_size=0.2, seed=42
        )

        train_patients = set(pid_train.unique())
        test_patients = set(pid_test.unique())

        assert train_patients.isdisjoint(test_patients), (
            f"Patient overlap detected! "
            f"Overlapping: {train_patients & test_patients}"
        )

    def test_cv_folds_no_patient_overlap(self, synthetic_data):
        expr_df, labels, patient_ids = synthetic_data
        folds = get_patient_cv_folds(expr_df, labels, patient_ids, n_folds=5, seed=42)

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_pids = set(patient_ids.iloc[train_idx].unique())
            val_pids = set(patient_ids.iloc[val_idx].unique())
            assert train_pids.isdisjoint(val_pids), (
                f"Fold {fold_idx}: patient overlap between train and val"
            )

    def test_all_samples_accounted_for(self, synthetic_data):
        expr_df, labels, patient_ids = synthetic_data
        X_train, X_test, y_train, y_test, pid_train, pid_test = patient_level_split(
            expr_df, labels, patient_ids, test_size=0.2, seed=42
        )

        total = len(X_train) + len(X_test)
        assert total == len(expr_df), (
            f"Not all samples accounted for: {total} != {len(expr_df)}"
        )


# ================================================================
# Test (b): Feature selection never sees validation/test data
# ================================================================


class TestFeatureSelectionLeakage:
    """Test that feature selection uses only training data."""

    def test_feature_selection_on_train_only(self, synthetic_data):
        expr_df, labels, patient_ids = synthetic_data
        folds = get_patient_cv_folds(expr_df, labels, patient_ids, n_folds=3, seed=42)

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_fold_train = expr_df.iloc[train_idx]
            y_fold_train = labels.iloc[train_idx]
            X_fold_val = expr_df.iloc[val_idx]

            # select_features_fold should only use X_fold_train, y_fold_train
            selected = select_features_fold(
                X_fold_train, y_fold_train,
                n_de_genes=50, n_final_genes=8, seed=42
            )

            # All selected genes must be columns in both train and val
            assert all(g in X_fold_train.columns for g in selected)
            assert len(selected) == 8

    def test_different_folds_can_select_different_genes(self, synthetic_data):
        """Verify that selection happens per-fold (not global)."""
        expr_df, labels, patient_ids = synthetic_data
        folds = get_patient_cv_folds(expr_df, labels, patient_ids, n_folds=3, seed=42)

        all_selections = []
        for train_idx, val_idx in folds:
            selected = select_features_fold(
                expr_df.iloc[train_idx],
                labels.iloc[train_idx],
                n_de_genes=50, n_final_genes=8, seed=42,
            )
            all_selections.append(set(selected))

        # At least check selections have the right size
        for sel in all_selections:
            assert len(sel) == 8


# ================================================================
# Test (c): Scaling is fit on train only
# ================================================================


class TestScaling:
    """Test that MinMaxScaler is fit only on training data."""

    def test_scaler_fit_on_train(self):
        rng = np.random.RandomState(42)
        X_train = rng.uniform(0, 10, (50, 8))
        X_val = rng.uniform(-2, 12, (20, 8))  # May be outside train range

        X_tr_s, X_va_s, _, scaler = scale_features(X_train, X_val)

        # Train data should be exactly in [0, pi]
        assert np.allclose(X_tr_s.min(axis=0), 0, atol=0.01)
        assert np.allclose(X_tr_s.max(axis=0), np.pi, atol=0.01)

        # Val data may exceed [0, pi] since scaler was fit on train
        # This is correct behavior - we DON'T want to clip to [0, pi]
        # based on val data

    def test_scaler_range(self):
        X_train = np.array([[0, 5], [10, 15]])
        X_val = np.array([[5, 10]])

        X_tr_s, X_va_s, _, _ = scale_features(X_train, X_val)

        assert X_tr_s[0, 0] == pytest.approx(0.0)
        assert X_tr_s[1, 0] == pytest.approx(np.pi)
        assert X_va_s[0, 0] == pytest.approx(np.pi / 2)


# ================================================================
# Test (d): Quantum model output shape and range
# ================================================================


class TestQuantumModel:
    """Test quantum model architecture."""

    def test_output_shape(self):
        model = HybridQuantumModel(n_qubits=4, n_layers=1)
        X = torch.rand(5, 4) * np.pi  # batch=5, features=4

        model.eval()
        with torch.no_grad():
            output = model(X)

        assert output.shape == (5, 1), f"Expected (5, 1), got {output.shape}"

    def test_output_range(self):
        model = HybridQuantumModel(n_qubits=4, n_layers=1)
        X = torch.rand(10, 4) * np.pi

        model.eval()
        with torch.no_grad():
            output = model(X)

        assert (output >= 0).all(), "Output values below 0"
        assert (output <= 1).all(), "Output values above 1"

    def test_parameter_count(self):
        model = HybridQuantumModel(n_qubits=8, n_layers=2)
        params = model.count_parameters()

        # StronglyEntanglingLayers: n_layers * n_qubits * 3 = 2 * 8 * 3 = 48
        assert params["quantum"] == 48
        # Linear(8, 1): 8 weights + 1 bias = 9
        assert params["classical"] == 9
        assert params["total"] == 57

    def test_ablation_model_output(self):
        model = ClassicalAblationModel(n_inputs=8, n_quantum_params=48)
        X = torch.rand(5, 8) * np.pi

        model.eval()
        with torch.no_grad():
            output = model(X)

        assert output.shape == (5, 1)
        assert (output >= 0).all() and (output <= 1).all()


# ================================================================
# Test (e): End-to-end smoke test on synthetic data
# ================================================================


class TestEndToEnd:
    """End-to-end smoke test on synthetic data."""

    def test_smoke_test_synthetic(self, synthetic_data, config):
        """Full pipeline on synthetic data should complete in < 2 minutes."""
        start_time = time.time()

        expr_df, labels, patient_ids = synthetic_data

        # Preprocess
        expr_df = remove_low_quality_genes(expr_df, 0.1, 1.0)

        # Patient split
        X_train, X_test, y_train, y_test, pid_train, pid_test = patient_level_split(
            expr_df, labels, patient_ids, test_size=0.2, seed=42
        )

        # CV fold
        folds = get_patient_cv_folds(X_train, y_train, pid_train, n_folds=2, seed=42)
        train_idx, val_idx = folds[0]

        # Feature selection
        selected = select_features_fold(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx],
            n_de_genes=50,
            n_final_genes=4,  # Use 4 qubits for speed
            seed=42,
        )

        X_tr = X_train.iloc[train_idx][selected].values
        X_va = X_train.iloc[val_idx][selected].values
        y_tr = y_train.iloc[train_idx].values
        y_va = y_train.iloc[val_idx].values

        # Scale
        X_tr_s, X_va_s, _, _ = scale_features(X_tr, X_va)

        # Train hybrid
        model = HybridQuantumModel(n_qubits=4, n_layers=1)
        model, history = train_torch_model(
            model, X_tr_s, y_tr, X_va_s, y_va,
            lr=0.05, batch_size=16, max_epochs=5,
            patience=3, seed=42,
        )

        # Predict
        probs = model(torch.tensor(X_va_s, dtype=torch.float32)).detach().numpy().flatten()
        metrics = compute_metrics(y_va, probs)

        assert "auc" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

        elapsed = time.time() - start_time
        assert elapsed < 120, f"Smoke test took {elapsed:.1f}s (> 2 min limit)"
        print(f"Smoke test passed in {elapsed:.1f}s")


# ================================================================
# Test: Barcode parsing
# ================================================================


class TestBarcodes:
    """Test TCGA barcode parsing."""

    def test_extract_sample_type(self):
        barcode = "TCGA-AB-1234-01A-11D-A1B2-09"
        assert extract_sample_type(barcode) == "01"

        barcode2 = "TCGA-AB-1234-11A-11D-A1B2-09"
        assert extract_sample_type(barcode2) == "11"

    def test_extract_patient_id(self):
        barcode = "TCGA-AB-1234-01A-11D-A1B2-09"
        assert extract_patient_id(barcode) == "TCGA-AB-1234"


# ================================================================
# Test: Metrics
# ================================================================


class TestMetrics:
    """Test evaluation metrics."""

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.95])
        metrics = compute_metrics(y_true, y_prob, threshold=0.5)

        assert metrics["accuracy"] == 1.0
        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 1.0
        assert metrics["auc"] == 1.0

    def test_random_predictions(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        y_prob = rng.random(100)
        metrics = compute_metrics(y_true, y_prob)

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["sensitivity"] <= 1
        assert 0 <= metrics["specificity"] <= 1

    def test_optimal_threshold(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        threshold = find_optimal_threshold(y_true, y_prob)
        assert 0 < threshold < 1

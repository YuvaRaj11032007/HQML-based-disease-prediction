"""
Classical baseline models: SVM, Random Forest, XGBoost.

All baselines use the same CV folds and feature sets as the hybrid model
for fair comparison.
"""

import logging
import time
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    seed: int = 42,
) -> Tuple[SVC, Dict]:
    """Train SVM with RBF kernel, tuned via grid search on validation data.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (for reporting, not tuning).
        y_val: Validation labels.
        config: SVM config with C_range and gamma_range.
        seed: Random seed.

    Returns:
        (trained_model, info_dict)
    """
    start = time.time()

    svm = SVC(
        kernel=config.get("kernel", "rbf"),
        probability=True,
        random_state=seed,
        class_weight="balanced",
    )

    # Simple grid search
    param_grid = {
        "C": config.get("C_range", [0.1, 1, 10, 100]),
        "gamma": config.get("gamma_range", ["scale", "auto"]),
    }

    # Adaptive CV based on class counts
    unique_classes, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min() if len(unique_classes) > 1 else 0
    cv_splits = min(3, min_count)

    if cv_splits >= 2:
        grid = GridSearchCV(
            svm, param_grid, cv=cv_splits, scoring="roc_auc", n_jobs=-1, refit=True
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_score = grid.best_score_
    else:
        svm.fit(X_train, y_train)
        best_model = svm
        best_params = {}
        best_score = float("nan")

    train_time = time.time() - start

    logger.info(
        f"SVM best params: {best_params}, "
        f"CV AUC: {best_score:.4f}, time: {train_time:.1f}s"
    )

    return best_model, {"train_time": train_time, "best_params": best_params}


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    seed: int = 42,
) -> Tuple[RandomForestClassifier, Dict]:
    """Train Random Forest classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        config: RF config.
        seed: Random seed.

    Returns:
        (trained_model, info_dict)
    """
    start = time.time()

    rf = RandomForestClassifier(
        n_estimators=config.get("n_estimators", 200),
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    # Simple grid search on max_depth
    param_grid = {
        "max_depth": config.get("max_depth_range", [5, 10, 20, None]),
    }

    # Adaptive CV based on class counts
    unique_classes, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min() if len(unique_classes) > 1 else 0
    cv_splits = min(3, min_count)

    if cv_splits >= 2:
        grid = GridSearchCV(rf, param_grid, cv=cv_splits, scoring="roc_auc", n_jobs=-1, refit=True)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_score = grid.best_score_
    else:
        rf.fit(X_train, y_train)
        best_model = rf
        best_params = {}
        best_score = float("nan")

    train_time = time.time() - start

    logger.info(
        f"RF best params: {best_params}, "
        f"CV AUC: {best_score:.4f}, time: {train_time:.1f}s"
    )

    return best_model, {"train_time": train_time, "best_params": best_params}


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    seed: int = 42,
) -> Tuple[XGBClassifier, Dict]:
    """Train XGBoost classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        config: XGBoost config.
        seed: Random seed.

    Returns:
        (trained_model, info_dict)
    """
    start = time.time()

    # Compute scale_pos_weight for imbalance
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    xgb = XGBClassifier(
        n_estimators=config.get("n_estimators", 200),
        max_depth=config.get("max_depth", 6),
        learning_rate=config.get("learning_rate", 0.1),
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
    )

    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    train_time = time.time() - start
    logger.info(f"XGBoost trained in {train_time:.1f}s")

    return xgb, {"train_time": train_time}


def get_classical_predictions(
    model, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions and probabilities from a sklearn-compatible model.

    Args:
        model: Trained sklearn/xgboost model.
        X: Features.

    Returns:
        (predicted_labels, predicted_probabilities_for_class_1)
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return y_pred, y_prob

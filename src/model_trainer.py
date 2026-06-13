"""
model_trainer.py
----------------
Handles model training for both fraud detection pipelines.

Models:
  - Logistic Regression  : interpretable baseline
  - Random Forest        : ensemble baseline
  - XGBoost              : primary ensemble model
  - LightGBM             : fast ensemble alternative

Both datasets share this module — the target column name differs:
  Fraud_Data.csv : 'class'
  creditcard.csv : 'Class'
"""

import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[model_trainer] Warning: xgboost not installed. XGBoost model unavailable.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[model_trainer] Warning: lightgbm not installed. LightGBM model unavailable.")


# ─────────────────────────────────────────────
# INPUT VALIDATION
# ─────────────────────────────────────────────

def _validate_inputs(X: pd.DataFrame, y: pd.Series, context: str = "") -> None:
    """Validate X and y before training or evaluation."""
    prefix = f"[{context}] " if context else ""

    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{prefix}X must be a pandas DataFrame, got {type(X).__name__}")
    if not isinstance(y, pd.Series):
        raise TypeError(f"{prefix}y must be a pandas Series, got {type(y).__name__}")
    if len(X) == 0:
        raise ValueError(f"{prefix}X is empty — no rows to train on")
    if len(y) == 0:
        raise ValueError(f"{prefix}y is empty — no target values")
    if len(X) != len(y):
        raise ValueError(
            f"{prefix}X and y length mismatch: X={len(X):,} rows, y={len(y):,} rows"
        )
    if X.isnull().any().any():
        null_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(
            f"{prefix}X contains NaN values in columns: {null_cols}. "
            f"Run data cleaning before training."
        )
    if y.isnull().any():
        raise ValueError(f"{prefix}y contains NaN values — check target column")

    n_classes = y.nunique()
    if n_classes < 2:
        raise ValueError(
            f"{prefix}y has only {n_classes} unique class(es). "
            f"Need at least 2 classes for classification."
        )


def _validate_column_names(X: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """
    Sanitize column names for LightGBM compatibility.
    LightGBM rejects special JSON characters: [ ] < > , { } " spaces
    Replaces them with underscores.
    """
    import re
    original = list(X.columns)
    cleaned  = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in original]

    if original != cleaned:
        changed = [(o, c) for o, c in zip(original, cleaned) if o != c]
        prefix = f"[{context}] " if context else ""
        print(f"{prefix}Cleaned {len(changed)} column name(s) for LightGBM compatibility:")
        for old, new in changed[:5]:
            print(f"  '{old}' → '{new}'")
        if len(changed) > 5:
            print(f"  ... and {len(changed) - 5} more")
        X = X.copy()
        X.columns = cleaned

    return X


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────

def get_logistic_regression(random_state: int = 42) -> LogisticRegression:
    """
    Logistic Regression baseline.

    Why Logistic Regression as baseline:
      - Fully interpretable — coefficients directly show feature impact
      - Fast to train — gives a quick performance floor
      - Linear decision boundary — if ensemble models barely beat it,
        the problem may be linearly separable and simpler models suffice
      - Regularization (C=0.1) prevents overfitting on high-dimensional
        one-hot encoded features

    max_iter=1000: fraud datasets often need more iterations to converge
    class_weight='balanced': additional handling for any residual imbalance
    """
    return LogisticRegression(
        C=0.1,
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        solver="lbfgs"
    )


def get_random_forest(n_estimators: int = 100,
                      max_depth: int = 10,
                      random_state: int = 42) -> RandomForestClassifier:
    """
    Random Forest ensemble model.

    Hyperparameters:
      n_estimators=100 : enough trees for stable predictions
      max_depth=10     : prevents overfitting on noisy fraud data
      class_weight='balanced_subsample': handles imbalance per tree
      min_samples_leaf=10: reduces noise from tiny leaf splits
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced_subsample",
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1
    )


def get_xgboost(n_estimators: int = 200,
                max_depth: int = 6,
                learning_rate: float = 0.05,
                random_state: int = 42):
    """
    XGBoost gradient boosting model.

    Why XGBoost as primary ensemble:
      - Handles imbalanced data well via scale_pos_weight
      - Built-in regularization (L1/L2) prevents overfitting
      - Feature importance natively available for SHAP analysis
      - Consistently strong performance on tabular fraud data
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "xgboost is not installed. Run: pip install xgboost"
        )
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0
    )


def get_lightgbm(n_estimators: int = 200,
                 max_depth: int = 6,
                 learning_rate: float = 0.05,
                 random_state: int = 42):
    """
    LightGBM gradient boosting model.

    Why include LightGBM:
      - Faster than XGBoost on large datasets (histogram-based splitting)
      - is_unbalance=True: native imbalance handling
      - Often matches XGBoost performance with less tuning

    Note: LightGBM does not support special JSON characters in column names.
          Column names are sanitized automatically in train_model().
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError(
            "lightgbm is not installed. Run: pip install lightgbm"
        )
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        is_unbalance=True,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1
    )


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_model(model,
                X_train: pd.DataFrame,
                y_train: pd.Series,
                model_name: str = "model") -> object:
    """
    Fit a model on training data and report training time.

    Automatically sanitizes column names for LightGBM compatibility.
    Validates inputs before training to catch issues early.
    """
    try:
        _validate_inputs(X_train, y_train, context=f"train_model/{model_name}")
    except (TypeError, ValueError) as e:
        raise type(e)(f"[train_model] Input validation failed: {e}") from e

    # Sanitize column names — required for LightGBM
    X_train = _validate_column_names(X_train, context=f"train_model/{model_name}")

    print(f"\n[train_model] Training {model_name}...")
    print(f"  X shape : {X_train.shape}")
    print(f"  y counts: {dict(y_train.value_counts().sort_index())}")

    start = time.time()
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        raise RuntimeError(
            f"[train_model] Training failed for {model_name}.\n"
            f"  Error: {e}\n"
            f"  Check: column names, NaN values, data types."
        ) from e

    elapsed = time.time() - start
    print(f"[train_model] {model_name} trained in {elapsed:.1f}s")
    return model


def save_model(model, path: str, model_name: str = "model") -> None:
    """Save a trained model to disk using joblib."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        print(f"[save_model] {model_name} saved to {path}")
    except Exception as e:
        raise IOError(
            f"[save_model] Failed to save {model_name} to '{path}': {e}"
        ) from e


def load_model(path: str) -> object:
    """Load a saved model from disk."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"[load_model] Model file not found: '{path}'\n"
            f"  Make sure modeling.ipynb has been run first."
        )
    try:
        model = joblib.load(path)
        print(f"[load_model] Loaded {type(model).__name__} from {path}")
        return model
    except Exception as e:
        raise IOError(
            f"[load_model] Failed to load model from '{path}': {e}"
        ) from e


# ─────────────────────────────────────────────
# CROSS VALIDATION
# ─────────────────────────────────────────────

def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "model",
    k: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Stratified K-Fold cross-validation.

    Why stratified:
      With severe class imbalance, random splits could put all fraud
      samples in one fold. Stratified splitting preserves the fraud
      class ratio in every fold, giving reliable metric estimates.

    Returns DataFrame with per-fold metrics and mean ± std summary.
    """
    try:
        _validate_inputs(X, y, context=f"cross_validate/{model_name}")
    except (TypeError, ValueError) as e:
        raise type(e)(f"[cross_validate_model] Input validation failed: {e}") from e

    if k < 2:
        raise ValueError(f"[cross_validate_model] k must be >= 2, got k={k}")

    # Check minimum samples per class for k folds
    min_class_count = y.value_counts().min()
    if min_class_count < k:
        raise ValueError(
            f"[cross_validate_model] Minority class has only {min_class_count} samples "
            f"but k={k} folds requested. Reduce k to at most {min_class_count}."
        )

    X = _validate_column_names(X, context=f"cross_validate/{model_name}")

    print(f"\n[cross_validate] Running {k}-fold stratified CV for {model_name}...")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr  = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr  = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        try:
            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
        except Exception as e:
            raise RuntimeError(
                f"[cross_validate_model] Fold {fold} failed for {model_name}: {e}"
            ) from e

        fold_results.append({
            "fold":      fold,
            "auc_pr":    round(average_precision_score(y_val, y_prob), 4),
            "f1":        round(f1_score(y_val, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_val, y_prob), 4),
            "precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_val, y_pred, zero_division=0), 4),
        })
        print(f"  Fold {fold}: AUC-PR={fold_results[-1]['auc_pr']:.4f} "
              f"F1={fold_results[-1]['f1']:.4f} "
              f"ROC-AUC={fold_results[-1]['roc_auc']:.4f}")

    results_df = pd.DataFrame(fold_results)

    print(f"\n[cross_validate] {model_name} — {k}-fold summary:")
    for col in ["auc_pr", "f1", "roc_auc", "precision", "recall"]:
        mean = results_df[col].mean()
        std  = results_df[col].std()
        print(f"  {col:12s}: {mean:.4f} ± {std:.4f}")

    return results_df


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dataset_name: str = "fraud",
    models_dir: str = "../models",
    run_cv: bool = True,
) -> Dict[str, Any]:
    """
    Train all models and return a dict of fitted models and CV results.

    Parameters
    ----------
    dataset_name : used for saved model filenames e.g. 'fraud' or 'creditcard'
    run_cv       : whether to run cross-validation (slow but thorough)

    Returns
    -------
    dict with keys: 'lr', 'rf', 'xgb', 'lgbm'
    Each value is a dict: {'name', 'model', 'cv_results'}
    """
    try:
        _validate_inputs(X_train, y_train, context="train_all_models")
    except (TypeError, ValueError) as e:
        raise type(e)(f"[train_all_models] {e}") from e

    model_defs = {
        "lr":   ("Logistic Regression", get_logistic_regression()),
        "rf":   ("Random Forest",       get_random_forest()),
    }
    if XGBOOST_AVAILABLE:
        model_defs["xgb"]  = ("XGBoost",   get_xgboost())
    else:
        print("[train_all_models] Skipping XGBoost — not installed")

    if LIGHTGBM_AVAILABLE:
        model_defs["lgbm"] = ("LightGBM",  get_lightgbm())
    else:
        print("[train_all_models] Skipping LightGBM — not installed")

    results = {}
    for key, (name, model) in model_defs.items():
        try:
            model = train_model(model, X_train, y_train, model_name=name)
            save_model(model, f"{models_dir}/{dataset_name}_{key}.pkl", model_name=name)
        except Exception as e:
            print(f"[train_all_models] ERROR training {name}: {e}")
            print(f"[train_all_models] Skipping {name} and continuing...")
            continue

        cv_results = None
        if run_cv:
            try:
                cv_results = cross_validate_model(
                    model.__class__(**model.get_params()),
                    X_train, y_train,
                    model_name=name
                )
            except Exception as e:
                print(f"[train_all_models] WARNING: CV failed for {name}: {e}")
                print(f"[train_all_models] Continuing without CV results for {name}")

        results[key] = {"name": name, "model": model, "cv_results": cv_results}

    if not results:
        raise RuntimeError(
            "[train_all_models] All models failed to train. "
            "Check your input data and dependencies."
        )

    return results
"""
transformation.py
-----------------
Handles data transformation steps:
  - Numerical feature scaling (StandardScaler / MinMaxScaler)
  - Categorical feature encoding (One-Hot Encoding)
  - Saving and loading fitted transformers for reuse at inference time

CRITICAL RULE: All transformers are fit on training data only.
               Never fit on test data — that would leak test statistics into training.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────
# SCALING
# ─────────────────────────────────────────────

def scale_numeric(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: List[str],
    method: str = "standard",
    save_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Fit a scaler on train_df and apply it to both train and test.

    Parameters
    ----------
    method : 'standard' — zero mean, unit variance (default)
             'minmax'   — scale to [0, 1]

    Returns
    -------
    (scaled_train_df, scaled_test_df, fitted_scaler)
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'standard' or 'minmax'.")

    available = [c for c in cols if c in train_df.columns]
    missing   = [c for c in cols if c not in train_df.columns]
    if missing:
        print(f"[scale_numeric] Warning: columns not found and skipped: {missing}")

    train_df = train_df.copy()
    test_df  = test_df.copy()

    train_df[available] = scaler.fit_transform(train_df[available])
    test_df[available]  = scaler.transform(test_df[available])

    print(f"[scale_numeric] Scaled {len(available)} columns using {method}Scaler")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"[scale_numeric] Scaler saved to {save_path}")

    return train_df, test_df, scaler


# ─────────────────────────────────────────────
# ENCODING
# ─────────────────────────────────────────────

def encode_categorical(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: List[str],
    drop: str = "first",
    save_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:

    available = [c for c in cols if c in train_df.columns]
    missing   = [c for c in cols if c not in train_df.columns]
    if missing:
        print(f"[encode_categorical] Warning: columns not found and skipped: {missing}")

    encoder = OneHotEncoder(drop=drop, sparse_output=False, handle_unknown="ignore")

    train_encoded = encoder.fit_transform(train_df[available])
    test_encoded  = encoder.transform(test_df[available])

    feature_names = encoder.get_feature_names_out(available)

    # Clean feature names — LightGBM rejects special JSON characters
    # [ ] < > , " { } spaces → replaced with underscores
    import re
    feature_names = [
        re.sub(r'[^A-Za-z0-9_]', '_', name)
        for name in feature_names
    ]

    train_enc_df = pd.DataFrame(train_encoded, columns=feature_names, index=train_df.index)
    test_enc_df  = pd.DataFrame(test_encoded,  columns=feature_names, index=test_df.index)

    train_df = train_df.drop(columns=available).join(train_enc_df)
    test_df  = test_df.drop(columns=available).join(test_enc_df)

    print(f"[encode_categorical] Encoded {len(available)} columns → {len(feature_names)} new columns")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoder, save_path)
        print(f"[encode_categorical] Encoder saved to {save_path}")

    return train_df, test_df, encoder


# ─────────────────────────────────────────────
# DROP COLUMNS
# ─────────────────────────────────────────────

def drop_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Drop columns that should not be used as model features."""
    to_drop = [c for c in cols if c in df.columns]
    skipped = [c for c in cols if c not in df.columns]
    if skipped:
        print(f"[drop_columns] Skipped (not found): {skipped}")
    df = df.drop(columns=to_drop)
    print(f"[drop_columns] Dropped: {to_drop}")
    return df


# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

def split_features_target(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target (y).
    Optionally drop columns that should not be model inputs.
    """
    df = df.copy()
    if drop_cols:
        existing = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=existing)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"[split_features_target] X: {X.shape} | y: {y.shape}")
    return X, y


# ─────────────────────────────────────────────
# LOAD SAVED TRANSFORMERS
# ─────────────────────────────────────────────

def load_transformer(path: str) -> object:
    """Load a previously saved scaler or encoder from disk."""
    transformer = joblib.load(path)
    print(f"[load_transformer] Loaded transformer from {path}")
    return transformer
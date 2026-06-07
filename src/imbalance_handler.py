"""
imbalance_handler.py
--------------------
Handles class imbalance for fraud detection datasets.

Techniques:
  - SMOTE          : Synthetic Minority Over-sampling Technique
  - Undersampling  : Randomly remove majority class rows
  - SMOTETomek     : SMOTE + Tomek links cleaning (recommended default)

CRITICAL RULE:
  Resampling is ALWAYS applied to training data only.
  The test set must remain untouched to reflect real-world distribution.
"""

import pandas as pd
import numpy as np
from typing import Tuple

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek


# ─────────────────────────────────────────────
# INSPECTION
# ─────────────────────────────────────────────

def class_distribution(y: pd.Series, label: str = "") -> pd.DataFrame:
    """Print and return class distribution of a target Series."""
    counts = y.value_counts().sort_index()
    pct    = (counts / len(y) * 100).round(2)
    df     = pd.DataFrame({
        "class": counts.index,
        "count": counts.values,
        "pct":   pct.values
    })
    header = f"Class distribution {label}".strip()
    print(f"\n[{header}]")
    print(df.to_string(index=False))
    ratio = counts.min() / counts.max()
    print(f"  Imbalance ratio (minority/majority): {ratio:.4f}")
    return df


# ─────────────────────────────────────────────
# RESAMPLING TECHNIQUES
# ─────────────────────────────────────────────

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to the training set.

    What SMOTE does:
      For each minority sample, finds k nearest neighbors and creates
      synthetic samples along the line segments between them.

    Best when: enough minority samples exist (>50) and class boundary is clear.
    Limitation: can create noise if minority samples overlap with majority.
    """
    print(f"\n[apply_smote] Applying SMOTE (k_neighbors={k_neighbors})...")
    class_distribution(y_train, "BEFORE SMOTE")

    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)

    class_distribution(y_res, "AFTER SMOTE")
    print(f"[apply_smote] Shape: {X_train.shape} → {X_res.shape}")
    return X_res, y_res


def apply_undersampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply random undersampling to the majority class.

    Best when: dataset is very large and SMOTE is too slow.
    Limitation: discards potentially useful majority class data.

    Parameters
    ----------
    sampling_strategy : desired minority/majority ratio after resampling.
                        0.5 = minority will be 50% of majority count.
    """
    print(f"\n[apply_undersampling] Applying Random Undersampling...")
    class_distribution(y_train, "BEFORE undersampling")

    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = rus.fit_resample(X_train, y_train)

    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)

    class_distribution(y_res, "AFTER undersampling")
    print(f"[apply_undersampling] Shape: {X_train.shape} → {X_res.shape}")
    return X_res, y_res


def apply_smote_tomek(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTETomek — SMOTE oversampling + Tomek links cleaning.

    What Tomek links are:
      A pair of samples from opposite classes that are each other's
      nearest neighbor. Removing the majority sample in each pair
      cleans the decision boundary.

    Why this is the recommended default:
      SMOTE adds synthetic minority samples, then Tomek links removal
      cleans ambiguous boundary samples from the majority class.
      Result: better class separation, less noise than SMOTE alone.
      Especially effective for financial fraud with overlapping distributions.
    """
    print(f"\n[apply_smote_tomek] Applying SMOTETomek...")
    class_distribution(y_train, "BEFORE SMOTETomek")

    smt = SMOTETomek(random_state=random_state)
    X_res, y_res = smt.fit_resample(X_train, y_train)

    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)

    class_distribution(y_res, "AFTER SMOTETomek")
    print(f"[apply_smote_tomek] Shape: {X_train.shape} → {X_res.shape}")
    return X_res, y_res


# ─────────────────────────────────────────────
# MAIN HANDLER
# ─────────────────────────────────────────────

def handle_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    technique: str = "smote_tomek",
    random_state: int = 42,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Unified interface for all resampling techniques.

    Parameters
    ----------
    technique : 'smote'        — pure SMOTE oversampling
                'undersample'  — random undersampling
                'smote_tomek'  — SMOTETomek (default, recommended)
    """
    technique = technique.lower().strip()

    if technique == "smote":
        return apply_smote(X_train, y_train, random_state=random_state, **kwargs)
    elif technique in ("undersample", "undersampling"):
        return apply_undersampling(X_train, y_train, random_state=random_state, **kwargs)
    elif technique in ("smote_tomek", "smotetomek"):
        return apply_smote_tomek(X_train, y_train, random_state=random_state)
    else:
        raise ValueError(
            f"Unknown technique '{technique}'. "
            f"Choose from: 'smote', 'undersample', 'smote_tomek'"
        )
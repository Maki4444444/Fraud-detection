"""
data_cleaner.py
---------------
Handles all data cleaning steps:
  - Missing value inspection and handling
  - Duplicate removal
  - Data type corrections
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# INSPECTION
# ─────────────────────────────────────────────

def inspect(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print a full inspection summary of the dataframe."""
    print(f"\n{'='*55}")
    print(f"  INSPECTION: {name}")
    print(f"{'='*55}")
    print(f"  Shape        : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Duplicates   : {df.duplicated().sum():,}")
    print(f"\n  Dtypes:\n{df.dtypes.to_string()}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print(f"\n  Missing values: None")
    else:
        pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({"missing": missing, "pct": pct})
        print(f"\n  Missing values:\n{missing_df.to_string()}")
    print(f"{'='*55}\n")


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarizing missing values per column."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    result = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": pct
    }).query("missing_count > 0").sort_values("missing_pct", ascending=False)

    if result.empty:
        print("[missing_summary] No missing values found.")
    return result


# ─────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and report how many were dropped."""
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    print(f"[remove_duplicates] Dropped {dropped:,} duplicate rows ({before:,} → {len(df):,})")
    return df


def handle_missing(df: pd.DataFrame, strategy: dict = None) -> pd.DataFrame:
    """
    Handle missing values column by column.

    strategy: dict mapping column name to action string.
      Actions:
        'drop'   — drop rows where this column is null
        'mean'   — fill with column mean
        'median' — fill with column median
        'mode'   — fill with column mode
        'ffill'  — forward fill
        'bfill'  — backward fill
        '<value>'— fill with a literal value

    If strategy is None, drops rows with any missing values.
    """
    if strategy is None:
        before = len(df)
        df = df.dropna()
        print(f"[handle_missing] Dropped {before - len(df):,} rows with any null")
        return df

    for col, action in strategy.items():
        if col not in df.columns:
            print(f"[handle_missing] Warning: '{col}' not found, skipping")
            continue

        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue

        if action == "drop":
            df = df[df[col].notna()]
            print(f"[handle_missing] '{col}': dropped {n_missing:,} rows")
        elif action == "mean":
            fill = df[col].mean()
            df[col] = df[col].fillna(fill)
            print(f"[handle_missing] '{col}': filled {n_missing:,} nulls with mean={fill:.4f}")
        elif action == "median":
            fill = df[col].median()
            df[col] = df[col].fillna(fill)
            print(f"[handle_missing] '{col}': filled {n_missing:,} nulls with median={fill:.4f}")
        elif action == "mode":
            fill = df[col].mode()[0]
            df[col] = df[col].fillna(fill)
            print(f"[handle_missing] '{col}': filled {n_missing:,} nulls with mode='{fill}'")
        elif action == "ffill":
            df[col] = df[col].ffill()
            print(f"[handle_missing] '{col}': forward-filled {n_missing:,} nulls")
        elif action == "bfill":
            df[col] = df[col].bfill()
            print(f"[handle_missing] '{col}': backward-filled {n_missing:,} nulls")
        else:
            df[col] = df[col].fillna(action)
            print(f"[handle_missing] '{col}': filled {n_missing:,} nulls with '{action}'")

    return df


# ─────────────────────────────────────────────
# DATA TYPE CORRECTIONS
# ─────────────────────────────────────────────

def fix_dtypes_fraud(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix data types specific to Fraud_Data.csv:
      - signup_time, purchase_time → datetime
      - user_id, device_id         → string (identifiers, not numbers)
      - class                       → int (target)
    """
    for col in ["signup_time", "purchase_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            print(f"[fix_dtypes] '{col}' → datetime")

    for col in ["user_id", "device_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
            print(f"[fix_dtypes] '{col}' → string")

    if "class" in df.columns:
        df["class"] = df["class"].astype(int)
        print(f"[fix_dtypes] 'class' → int")

    return df


def fix_dtypes_creditcard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix data types for creditcard.csv.
    All V1–V28 and Amount are already float; ensure Class is int.
    """
    if "Class" in df.columns:
        df["Class"] = df["Class"].astype(int)
        print(f"[fix_dtypes] 'Class' → int")
    return df


# ─────────────────────────────────────────────
# COMBINED PIPELINES
# ─────────────────────────────────────────────

def clean_fraud_data(df: pd.DataFrame, missing_strategy: dict = None) -> pd.DataFrame:
    """Full cleaning pipeline for Fraud_Data.csv."""
    print("\n--- Cleaning Fraud_Data ---")
    df = remove_duplicates(df)
    df = handle_missing(df, strategy=missing_strategy)
    df = fix_dtypes_fraud(df)
    print(f"[clean_fraud_data] Done. Final shape: {df.shape}")
    return df


def clean_creditcard(df: pd.DataFrame, missing_strategy: dict = None) -> pd.DataFrame:
    """Full cleaning pipeline for creditcard.csv."""
    print("\n--- Cleaning creditcard ---")
    df = remove_duplicates(df)
    df = handle_missing(df, strategy=missing_strategy)
    df = fix_dtypes_creditcard(df)
    print(f"[clean_creditcard] Done. Final shape: {df.shape}")
    return df
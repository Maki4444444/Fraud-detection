"""
data_loader.py
--------------
Responsible for loading raw datasets from disk.
Paths are resolved relative to the project root automatically.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"


def load_fraud_data(path: str = None) -> pd.DataFrame:
    """Load the e-commerce fraud dataset."""
    file_path = Path(path) if path else RAW_DIR / "Fraud_Data.csv"
    df = pd.read_csv(file_path)
    print(f"[fraud_data] Loaded {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def load_ip_country(path: str = None) -> pd.DataFrame:
    """Load the IP address to country mapping dataset."""
    file_path = Path(path) if path else RAW_DIR / "IpAddress_to_Country.csv"
    df = pd.read_csv(file_path)
    print(f"[ip_country] Loaded {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def load_creditcard(path: str = None) -> pd.DataFrame:
    """Load the bank credit card transactions dataset."""
    file_path = Path(path) if path else RAW_DIR / "creditcard.csv"
    df = pd.read_csv(file_path)
    print(f"[creditcard] Loaded {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def load_all(raw_dir: str = None) -> dict:
    """
    Load all three datasets at once.
    Returns a dict with keys: 'fraud', 'ip_country', 'creditcard'
    """
    base = Path(raw_dir) if raw_dir else RAW_DIR
    return {
        "fraud":      load_fraud_data(base / "Fraud_Data.csv"),
        "ip_country": load_ip_country(base / "IpAddress_to_Country.csv"),
        "creditcard": load_creditcard(base / "creditcard.csv"),
    }
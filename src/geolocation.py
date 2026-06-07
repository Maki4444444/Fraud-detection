"""
geolocation.py
--------------
Handles IP address geolocation enrichment for Fraud_Data.csv:
  - Convert IP addresses to integers
  - Range-based merge with IpAddress_to_Country.csv
  - Fraud pattern analysis by country

Key finding about the data format:
  Fraud_Data.csv stores IPs as floats e.g. 732758368.7997
  IpAddress_to_Country.csv stores bounds as plain integers e.g. 16777216
  So conversion is just: int(round(float_ip))
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# IP CONVERSION
# ─────────────────────────────────────────────

def ip_to_int(ip) -> int:
    """
    Convert an IP value to a 32-bit integer.

    Fraud_Data.csv stores IPs as floats like 732758368.7997.
    The decimal part is floating point noise — we round and cast to int.

    Also handles dotted string format '192.168.1.1' as a fallback.
    """
    try:
        if ip is None or (isinstance(ip, float) and np.isnan(ip)):
            return np.nan

        # Already numeric (float or int) — the format in Fraud_Data.csv
        if isinstance(ip, (int, float)):
            return int(round(ip))

        # String — try numeric string first e.g. '732758368.7997'
        ip_str = str(ip).strip()
        try:
            return int(round(float(ip_str)))
        except ValueError:
            pass

        # Last resort: dotted decimal format '192.168.1.1'
        parts = ip_str.split(".")
        if len(parts) == 4:
            return sum(int(p) * (256 ** (3 - i)) for i, p in enumerate(parts))

        return np.nan

    except Exception:
        return np.nan


def convert_ip_column(df: pd.DataFrame, ip_col: str = "ip_address") -> pd.DataFrame:
    """
    Add an integer IP column to the dataframe.
    New column will be named '{ip_col}_int'.
    """
    int_col = f"{ip_col}_int"
    df = df.copy()
    df[int_col] = df[ip_col].apply(ip_to_int)

    n_failed = df[int_col].isna().sum()
    if n_failed > 0:
        print(f"[convert_ip_column] Warning: {n_failed:,} IPs could not be converted")
    else:
        print(f"[convert_ip_column] All {len(df):,} IPs converted successfully")

    print(f"[convert_ip_column] Sample:")
    print(df[[ip_col, int_col]].head(3).to_string(index=False))

    return df


# ─────────────────────────────────────────────
# RANGE-BASED COUNTRY MERGE
# ─────────────────────────────────────────────

def merge_country(
    fraud_df: pd.DataFrame,
    ip_country_df: pd.DataFrame,
    fraud_ip_int_col: str = "ip_address_int",
    lower_col: str = "lower_bound_ip_address",
    upper_col: str = "upper_bound_ip_address",
    country_col: str = "country",
) -> pd.DataFrame:
    """
    Enrich fraud_df with country information via a range-based IP lookup.

    Why range-based and not a direct join:
      IpAddress_to_Country defines ranges: every IP from X to Y belongs
      to a given country. No single key to join on — we find which range
      each IP falls into using binary search.
    """
    ip_country_sorted = ip_country_df.copy()
    ip_country_sorted[lower_col] = pd.to_numeric(
        ip_country_sorted[lower_col], errors="coerce"
    )
    ip_country_sorted[upper_col] = pd.to_numeric(
        ip_country_sorted[upper_col], errors="coerce"
    )
    ip_country_sorted = ip_country_sorted.dropna(subset=[lower_col, upper_col])
    ip_country_sorted = ip_country_sorted.sort_values(lower_col).reset_index(drop=True)

    lower_bounds = ip_country_sorted[lower_col].to_numpy(dtype=np.int64)
    upper_bounds = ip_country_sorted[upper_col].to_numpy(dtype=np.int64)
    countries    = ip_country_sorted[country_col].values

    def lookup_country(ip_int):
        if ip_int is None or (isinstance(ip_int, float) and np.isnan(ip_int)):
            return "Unknown"
        ip_int = int(ip_int)
        idx = np.searchsorted(lower_bounds, ip_int, side="right") - 1
        if idx >= 0 and lower_bounds[idx] <= ip_int <= upper_bounds[idx]:
            return countries[idx]
        return "Unknown"

    print(f"[merge_country] Looking up countries for {len(fraud_df):,} transactions...")
    fraud_df = fraud_df.copy()
    fraud_df[country_col] = fraud_df[fraud_ip_int_col].apply(lookup_country)

    n_unknown = (fraud_df[country_col] == "Unknown").sum()
    n_matched = len(fraud_df) - n_unknown
    print(
        f"[merge_country] Matched: {n_matched:,} | Unknown: {n_unknown:,} "
        f"({n_unknown / len(fraud_df) * 100:.1f}%)"
    )

    return fraud_df


# ─────────────────────────────────────────────
# COMBINED PIPELINE
# ─────────────────────────────────────────────

def enrich_with_country(
    fraud_df: pd.DataFrame,
    ip_country_df: pd.DataFrame,
    ip_col: str = "ip_address",
) -> pd.DataFrame:
    """
    Full geolocation pipeline:
      1. Convert IP float values to integers (732758368.7997 → 732758368)
      2. Merge with country lookup via binary search on IP ranges
    Returns enriched DataFrame with 'country' column added.
    """
    fraud_df = convert_ip_column(fraud_df, ip_col=ip_col)
    fraud_df = merge_country(
        fraud_df,
        ip_country_df,
        fraud_ip_int_col=f"{ip_col}_int",
    )
    return fraud_df


# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────

def fraud_by_country_summary(
    df: pd.DataFrame,
    country_col: str = "country",
    target_col: str = "class",
    min_transactions: int = 50,
) -> pd.DataFrame:
    """
    Return a summary table of fraud counts and rates per country.
    Filters out countries with fewer than min_transactions.
    """
    grouped = df.groupby(country_col)[target_col].agg(
        total="count",
        fraud_count="sum"
    ).reset_index()

    grouped["fraud_rate_pct"] = (
        grouped["fraud_count"] / grouped["total"] * 100
    ).round(2)
    grouped = grouped[grouped["total"] >= min_transactions]
    grouped = grouped.sort_values("fraud_rate_pct", ascending=False)

    print(
        f"\n[fraud_by_country] Top 10 countries by fraud rate "
        f"(min {min_transactions} transactions):"
    )
    print(grouped.head(10).to_string(index=False))

    return grouped
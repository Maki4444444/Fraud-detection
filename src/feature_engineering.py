"""
feature_engineering.py
-----------------------
Creates new predictive features for Fraud_Data.csv.

Key dataset findings that shaped feature design:
  1. Each user_id appears exactly once — per-user transaction frequency
     is always 0 and carries no signal. Frequency must be computed at
     device level instead.

  2. Devices used more than once have a 52.46% fraud rate vs 3.04% for
     single-use devices — a 17x difference. Device reuse is one of the
     strongest fraud signals in the dataset.

  3. 54% of fraud transactions occur within 1 hour of account creation
     vs 0% for legitimate users — time_since_signup is the dominant feature.

Features created:
  - hour_of_day           : hour extracted from purchase_time (0-23)
  - day_of_week           : day extracted from purchase_time (0=Mon, 6=Sun)
  - time_since_signup_sec : seconds between signup_time and purchase_time
  - time_since_signup_hours: time_since_signup_sec / 3600
  - device_reuse_count    : how many times this device appears in the dataset
  - device_reused         : binary flag — 1 if device appears more than once
  - transaction_count_1h  : transactions by same device in past 1 hour
  - transaction_count_24h : transactions by same device in past 24 hours
  - transaction_velocity  : transaction_count_24h / (time_since_signup_hours + 1)
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# TIME FEATURES
# ─────────────────────────────────────────────

def add_time_features(df: pd.DataFrame,
                      purchase_col: str = "purchase_time") -> pd.DataFrame:
    """
    Extract hour_of_day and day_of_week from purchase timestamp.

    Why these matter:
      - Fraud rate varies by hour — elevated during business hours in this dataset
      - Weekend fraud rate (Fri–Sun) is higher than mid-week (Tue–Wed)
      - day_of_week correlation with fraud: 0.0189 (statistically significant)
      - hour_of_day correlation with fraud: 0.0020 (not significant — weak signal)
    """
    if purchase_col not in df.columns:
        raise ValueError(f"Column '{purchase_col}' not found in DataFrame")

    df = df.copy()
    dt = pd.to_datetime(df[purchase_col])
    df["hour_of_day"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek  # 0 = Monday, 6 = Sunday

    print(f"[add_time_features] Added: hour_of_day, day_of_week")
    return df


def add_time_since_signup(df: pd.DataFrame,
                           signup_col: str = "signup_time",
                           purchase_col: str = "purchase_time") -> pd.DataFrame:
    """
    Compute time elapsed between account creation and purchase.

    Why this matters:
      Confirmed finding: 54% of fraud transactions occur within 1 hour
      of account creation vs 0.0% for legitimate users.
      Fraud median: 0.0003 hours (~1 second)
      Legit median: 1,443 hours (~60 days)
      Mann-Whitney p-value: 0.000000 — distributions are significantly different.
      Correlation with fraud: -0.2579 — strongest single feature in dataset.

    Adds:
      - time_since_signup_sec   : raw seconds (kept for reference)
      - time_since_signup_hours : in hours (used for modeling and velocity)

    Negative values (purchase before signup = data error) are clamped to 0.
    """
    df = df.copy()
    signup   = pd.to_datetime(df[signup_col])
    purchase = pd.to_datetime(df[purchase_col])
    delta    = (purchase - signup).dt.total_seconds()

    n_negative = (delta < 0).sum()
    if n_negative > 0:
        print(f"[add_time_since_signup] Warning: {n_negative:,} rows have "
              f"purchase before signup — clamped to 0")
        delta = delta.clip(lower=0)

    df["time_since_signup_sec"]   = delta
    df["time_since_signup_hours"] = (delta / 3600).round(4)

    print(f"[add_time_since_signup] Added: time_since_signup_sec, time_since_signup_hours")
    print(f"  Median time since signup: {delta.median() / 3600:.2f} hours")
    return df


# ─────────────────────────────────────────────
# DEVICE REUSE FEATURES
# ─────────────────────────────────────────────

def add_device_reuse(df: pd.DataFrame,
                     device_col: str = "device_id") -> pd.DataFrame:
    """
    Add device reuse features.

    Key finding: devices used more than once have a 52.46% fraud rate
    vs 3.04% for single-use devices — a 17x difference.

    Adds:
      - device_reuse_count : how many times this device appears in the dataset
      - device_reused      : binary flag — 1 if device appears more than once
    """
    df = df.copy()
    counts = df[device_col].value_counts()
    df["device_reuse_count"] = df[device_col].map(counts)
    df["device_reused"]      = (df["device_reuse_count"] > 1).astype(int)

    reused_fraud = df[df["device_reused"] == 1]["class"].mean()
    single_fraud = df[df["device_reused"] == 0]["class"].mean()

    print(f"[add_device_reuse] Added: device_reuse_count, device_reused")
    print(f"  Reused device fraud rate : {reused_fraud:.4f} ({reused_fraud*100:.2f}%)")
    print(f"  Single device fraud rate : {single_fraud:.4f} ({single_fraud*100:.2f}%)")
    print(f"  Fraud rate multiplier    : {reused_fraud/single_fraud:.1f}x")

    return df


# ─────────────────────────────────────────────
# TRANSACTION FREQUENCY & VELOCITY
# ─────────────────────────────────────────────

def add_transaction_frequency(df: pd.DataFrame,
                               user_col: str = "device_id",
                               purchase_col: str = "purchase_time",
                               windows: list = [1, 24]) -> pd.DataFrame:
    """
    For each transaction, count how many times the same device
    transacted in the preceding N hours.

    Why device_id and not user_id:
      Each user_id appears exactly once in Fraud_Data.csv — every
      transaction belongs to a unique user, so per-user frequency
      is always 0 and carries zero signal.

      Fraudsters reuse the same physical device across multiple fake
      accounts. Device-level frequency captures this pattern:
      multiple transactions from the same device in a short window
      is a strong indicator of coordinated fraud.

    Parameters
    ----------
    user_col : grouping column — defaults to 'device_id'
    windows  : list of time windows in hours — default [1, 24]

    Adds columns:
      transaction_count_1h  — device transaction count in past 1 hour
      transaction_count_24h — device transaction count in past 24 hours

    Note: sorts by purchase_time internally and resets index.
    """
    df = df.copy()
    df[purchase_col] = pd.to_datetime(df[purchase_col])
    df = df.sort_values([user_col, purchase_col]).reset_index(drop=True)

    for window_hours in windows:
        col_name = f"transaction_count_{window_hours}h"
        counts = []

        for device, group in df.groupby(user_col, sort=False):
            times = group[purchase_col]
            window_td = pd.Timedelta(hours=window_hours)
            device_counts = []
            for i, t in enumerate(times):
                window_start = t - window_td
                cnt = ((times[:i] >= window_start) & (times[:i] < t)).sum()
                device_counts.append(cnt)
            counts.extend(device_counts)

        df[col_name] = counts
        non_zero = (df[col_name] > 0).sum()
        print(f"[add_transaction_frequency] Added: {col_name} "
              f"(non-zero rows: {non_zero:,} = {non_zero/len(df)*100:.1f}%)")

    return df


def add_transaction_velocity(df: pd.DataFrame,
                              count_col: str = "transaction_count_24h",
                              hours_col: str = "time_since_signup_hours") -> pd.DataFrame:
    """
    Compute transaction velocity: device transactions in 24h relative to account age.

    Formula:
      transaction_velocity = transaction_count_24h / (time_since_signup_hours + 1)

    Why +1:
      Prevents division by zero for brand-new accounts (time_since_signup = 0).
      Ensures accounts with 0 hours since signup are treated as 1 hour old,
      which slightly dampens the velocity score for the newest accounts —
      a conservative choice that avoids infinite values.

    Why this is meaningful:
      A high transaction count is far more suspicious on a 1-hour-old account
      than on a 2-year-old account. This ratio captures that joint signal:
      high velocity = many device transactions on a very new account.

    Example:
      Fraudster  : count_24h=5, signup_hours=0.5  → velocity = 5/1.5  = 3.33
      Legitimate : count_24h=5, signup_hours=8760 → velocity = 5/8761 = 0.00057
    """
    if count_col not in df.columns or hours_col not in df.columns:
        raise ValueError(
            f"Required columns missing. Need '{count_col}' and '{hours_col}'. "
            f"Run add_transaction_frequency() and add_time_since_signup() first."
        )
    df = df.copy()
    df["transaction_velocity"] = df[count_col] / (df[hours_col] + 1)

    if "class" in df.columns:
        fraud_mean = df[df["class"] == 1]["transaction_velocity"].mean()
        legit_mean = df[df["class"] == 0]["transaction_velocity"].mean()
        print(f"[add_transaction_velocity] Added: transaction_velocity")
        print(f"  Fraud mean velocity : {fraud_mean:.4f}")
        print(f"  Legit mean velocity : {legit_mean:.4f}")
    else:
        print(f"[add_transaction_velocity] Added: transaction_velocity")

    return df


# ─────────────────────────────────────────────
# COMBINED PIPELINE
# ─────────────────────────────────────────────

def engineer_all_features(df: pd.DataFrame,
                           frequency_windows: list = [1, 24],
                           user_col: str = "device_id") -> pd.DataFrame:
    """
    Run the full feature engineering pipeline for Fraud_Data.csv.

    Parameters
    ----------
    frequency_windows : time windows in hours for transaction frequency
    user_col          : grouping column for frequency features.
                        Defaults to 'device_id' — NOT 'user_id'.
                        Reason: each user_id appears exactly once in this
                        dataset making per-user frequency always 0.
                        device_id captures cross-account device reuse
                        which is a confirmed strong fraud signal (17x
                        higher fraud rate on reused devices).

    Pipeline order (order matters — later steps depend on earlier ones):
      1. add_time_features()        needs: purchase_time
      2. add_time_since_signup()    needs: signup_time, purchase_time
      3. add_device_reuse()         needs: device_id
      4. add_transaction_frequency()needs: device_id, purchase_time
      5. add_transaction_velocity() needs: output of steps 2 and 4
    """
    print("\n--- Feature Engineering Pipeline ---")
    print(f"[engineer_all_features] Frequency grouping column: '{user_col}'")

    df = add_time_features(df)
    df = add_time_since_signup(df)
    df = add_device_reuse(df)
    df = add_transaction_frequency(df, user_col=user_col, windows=frequency_windows)
    df = add_transaction_velocity(df)

    new_cols = [
        "hour_of_day",
        "day_of_week",
        "time_since_signup_sec",
        "time_since_signup_hours",
        "device_reuse_count",
        "device_reused",
        *[f"transaction_count_{w}h" for w in frequency_windows],
        "transaction_velocity",
    ]

    print(f"\n[engineer_all_features] Done. New columns added:")
    for col in new_cols:
        status = "OK" if col in df.columns else "MISSING"
        print(f"  [{status}] {col}")

    return df


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

def feature_summary(df: pd.DataFrame, new_cols: list) -> pd.DataFrame:
    """
    Print basic statistics for newly engineered features,
    broken down by fraud class if 'class' column exists.
    """
    available = [c for c in new_cols if c in df.columns]
    missing   = [c for c in new_cols if c not in df.columns]

    if missing:
        print(f"[feature_summary] Warning: columns not found: {missing}")

    summary = df[available].describe().T
    print("\n[feature_summary] Engineered feature statistics:")
    print(summary.to_string())

    if "class" in df.columns:
        print("\n[feature_summary] Mean by class (0=legit, 1=fraud):")
        print(df.groupby("class")[available].mean().T.to_string())

    return summary
"""
eda.py
------
Reusable EDA functions for both Fraud_Data and creditcard datasets.
Covers:
  - Univariate distributions
  - Bivariate analysis vs target
  - Class imbalance quantification
  - Correlation heatmaps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import List, Optional

# ── Global style ──────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
# List palette — works regardless of whether class values are int or string
FRAUD_PALETTE = ["#4C9BE8", "#E8524C"]  # index 0 = legit (blue), index 1 = fraud (red)


# ─────────────────────────────────────────────
# CLASS IMBALANCE
# ─────────────────────────────────────────────

def class_imbalance_report(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Print and return a summary table of class distribution."""
    counts = df[target_col].value_counts().sort_index()
    pct    = (counts / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "class":      counts.index,
        "count":      counts.values,
        "percentage": pct.values
    })
    print(f"\n[class_imbalance] '{target_col}' distribution:")
    print(summary.to_string(index=False))
    return summary


def plot_class_imbalance(df: pd.DataFrame, target_col: str, title: str = "") -> None:
    """Bar chart showing class distribution with counts and percentages."""
    counts = df[target_col].value_counts().sort_index()
    labels = [f"Class {i}" for i in counts.index]
    pct    = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    # Use index position into FRAUD_PALETTE list — avoids dict key type issues
    colors = [FRAUD_PALETTE[i] if i < len(FRAUD_PALETTE) else "#999"
              for i in range(len(counts))]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)

    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.01,
                f"{bar.get_height():,}\n({p:.1f}%)",
                ha="center", va="bottom", fontsize=10)

    ax.set_title(title or f"Class Distribution — {target_col}", fontsize=13)
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# UNIVARIATE
# ─────────────────────────────────────────────

def plot_numeric_distributions(df: pd.DataFrame,
                                cols: List[str],
                                n_cols: int = 3) -> None:
    """Plot histograms with KDE for a list of numeric columns."""
    n_rows = -(-len(cols) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i],
                     color=FRAUD_PALETTE[0], edgecolor="white")
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("")

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numeric Feature Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_categorical_counts(df: pd.DataFrame,
                             cols: List[str],
                             n_cols: int = 2,
                             top_n: int = 10) -> None:
    """Bar charts for categorical columns, showing top_n categories."""
    n_rows = -(-len(cols) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        top = df[col].value_counts().head(top_n)
        sns.barplot(x=top.values, y=top.index.astype(str),
                    ax=axes[i], palette="Blues_d")
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("Count")

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Categorical Feature Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# BIVARIATE vs TARGET
# ─────────────────────────────────────────────

def plot_numeric_vs_target(df: pd.DataFrame,
                            cols: List[str],
                            target_col: str,
                            n_cols: int = 3) -> None:
    """
    Box plots comparing numeric feature distributions
    between fraud and non-fraud classes.
    """
    n_rows = -(-len(cols) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    # Convert target to string so seaborn hue mapping is consistent
    df = df.copy()
    df[target_col] = df[target_col].astype(str)
    unique_vals = sorted(df[target_col].unique())
    palette = {v: FRAUD_PALETTE[i] if i < len(FRAUD_PALETTE) else "#999"
               for i, v in enumerate(unique_vals)}

    for i, col in enumerate(cols):
        sns.boxplot(data=df, x=target_col, y=col,
                    palette=palette, ax=axes[i],
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel(target_col)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Numeric Features vs {target_col}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_categorical_vs_target(df: pd.DataFrame,
                                cols: List[str],
                                target_col: str,
                                n_cols: int = 2,
                                top_n: int = 10) -> None:
    """Horizontal bar charts showing fraud rate per category."""
    n_rows = -(-len(cols) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        top_cats = df[col].value_counts().head(top_n).index
        subset = df[df[col].isin(top_cats)]
        fraud_rate = (
            subset.groupby(col)[target_col].mean() * 100
        ).sort_values(ascending=False)

        sns.barplot(x=fraud_rate.values, y=fraud_rate.index.astype(str),
                    ax=axes[i], palette="Reds_d")
        axes[i].set_title(f"Fraud rate by {col}", fontsize=11)
        axes[i].set_xlabel("Fraud rate (%)")

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Categorical Features vs {target_col} (fraud rate %)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# CORRELATION
# ─────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame,
                              target_col: Optional[str] = None,
                              figsize: tuple = (14, 10)) -> None:
    """
    Full correlation heatmap.
    If target_col is provided, also prints top correlations with target.
    """
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap", fontsize=13)
    plt.tight_layout()
    plt.show()

    if target_col and target_col in corr.columns:
        top_corr = (corr[target_col]
                    .drop(target_col)
                    .abs()
                    .sort_values(ascending=False)
                    .head(15))
        print(f"\nTop correlations with '{target_col}':")
        print(top_corr.to_string())


# ─────────────────────────────────────────────
# TIME-BASED
# ─────────────────────────────────────────────

def plot_fraud_by_hour(df: pd.DataFrame,
                       hour_col: str = "hour_of_day",
                       target_col: str = "class") -> None:
    """Line chart: fraud rate and transaction volume by hour of day."""
    grouped = df.groupby(hour_col)[target_col].agg(["sum", "count"])
    grouped["fraud_rate"] = grouped["sum"] / grouped["count"] * 100

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    ax1.bar(grouped.index, grouped["count"],
            color=FRAUD_PALETTE[0], alpha=0.5, label="Total transactions")
    ax2.plot(grouped.index, grouped["fraud_rate"],
             color=FRAUD_PALETTE[1], linewidth=2, marker="o", label="Fraud rate %")

    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Transaction Count")
    ax2.set_ylabel("Fraud Rate (%)")
    ax1.set_title("Transaction Volume and Fraud Rate by Hour of Day")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_fraud_by_day(df: pd.DataFrame,
                      day_col: str = "day_of_week",
                      target_col: str = "class") -> None:
    """Bar chart: fraud rate by day of week."""
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    grouped = df.groupby(day_col)[target_col].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([day_names[i] for i in grouped.index], grouped.values,
           color=FRAUD_PALETTE[1], alpha=0.8, edgecolor="white")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("Fraud Rate by Day of Week")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# COUNTRY ANALYSIS
# ─────────────────────────────────────────────

def plot_fraud_by_country(df: pd.DataFrame,
                           country_col: str = "country",
                           target_col: str = "class",
                           top_n: int = 15) -> None:
    """Horizontal bar chart: top N countries by fraud rate."""
    grouped = df.groupby(country_col)[target_col].agg(["sum", "count"])
    grouped["fraud_rate"] = grouped["sum"] / grouped["count"] * 100
    grouped = grouped[grouped["count"] >= 50]  # minimum sample threshold
    top = grouped.sort_values("fraud_rate", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top.index[::-1], top["fraud_rate"][::-1],
                   color=FRAUD_PALETTE[1], alpha=0.8, edgecolor="white")

    for bar, cnt in zip(bars, top["count"][::-1]):
        ax.text(bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"n={cnt:,}", va="center", fontsize=8, color="#555")

    ax.set_xlabel("Fraud Rate (%)")
    ax.set_title(f"Top {top_n} Countries by Fraud Rate (min 50 transactions)")
    plt.tight_layout()
    plt.show()
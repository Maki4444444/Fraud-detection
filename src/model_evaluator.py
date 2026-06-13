"""
model_evaluator.py
------------------
Handles model evaluation, metric computation, and comparison.

Primary metrics for fraud detection (imbalanced data):
  - AUC-PR  : Area Under Precision-Recall Curve — best metric for imbalanced
  - F1-Score: harmonic mean of precision and recall
  - Precision: of all flagged transactions, how many are real fraud?
  - Recall   : of all real fraud, how many did we catch?

Secondary metrics:
  - ROC-AUC  : standard but optimistic on imbalanced data
  - Confusion Matrix: raw counts of TP, FP, TN, FN

Why AUC-PR over ROC-AUC for fraud:
  ROC-AUC accounts for true negatives — but with 99%+ legitimate transactions,
  a model can get high ROC-AUC while still missing most fraud.
  AUC-PR only looks at the positive (fraud) class performance, making it
  a far more honest metric when the minority class is what matters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

from sklearn.metrics import (
    average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)

BLUE = "#4C9BE8"
RED  = "#E8524C"


# ─────────────────────────────────────────────
# INPUT VALIDATION
# ─────────────────────────────────────────────

def _validate_eval_inputs(model, X_test: pd.DataFrame,
                           y_test: pd.Series, context: str = "") -> None:
    """Validate inputs before evaluation."""
    prefix = f"[{context}] " if context else ""

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"{prefix}Model {type(model).__name__} does not have predict_proba(). "
            f"All models in this pipeline must support probability prediction."
        )
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            f"{prefix}X_test must be a pandas DataFrame, got {type(X_test).__name__}"
        )
    if not isinstance(y_test, pd.Series):
        raise TypeError(
            f"{prefix}y_test must be a pandas Series, got {type(y_test).__name__}"
        )
    if len(X_test) == 0:
        raise ValueError(f"{prefix}X_test is empty")
    if len(X_test) != len(y_test):
        raise ValueError(
            f"{prefix}X_test and y_test length mismatch: "
            f"X={len(X_test):,}, y={len(y_test):,}"
        )
    if X_test.isnull().any().any():
        null_cols = X_test.columns[X_test.isnull().any()].tolist()
        raise ValueError(
            f"{prefix}X_test contains NaN values in: {null_cols}"
        )
    if y_test.isnull().any():
        raise ValueError(f"{prefix}y_test contains NaN values")
    if y_test.nunique() < 2:
        raise ValueError(
            f"{prefix}y_test has only {y_test.nunique()} class(es) — need at least 2"
        )


def _validate_threshold(threshold: float) -> None:
    """Validate prediction threshold."""
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold must be float, got {type(threshold).__name__}")
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"threshold must be between 0 and 1 (exclusive), got {threshold}"
        )


def _clean_column_names(X: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names to match training — same logic as model_trainer."""
    import re
    X = X.copy()
    X.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in X.columns]
    return X


# ─────────────────────────────────────────────
# SINGLE MODEL EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate a trained model on the test set.
    Returns a dict of all metrics for easy comparison.
    """
    try:
        _validate_eval_inputs(model, X_test, y_test, context="evaluate_model")
        _validate_threshold(threshold)
    except (AttributeError, TypeError, ValueError) as e:
        raise type(e)(f"[evaluate_model] Validation failed: {e}") from e

    X_test = _clean_column_names(X_test)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        raise RuntimeError(
            f"[evaluate_model] predict_proba failed for {model_name}.\n"
            f"  This often means column names or count changed since training.\n"
            f"  Error: {e}"
        ) from e

    y_pred = (y_prob >= threshold).astype(int)

    try:
        metrics = {
            "model":     model_name,
            "auc_pr":    round(average_precision_score(y_test, y_prob), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        }
    except Exception as e:
        raise RuntimeError(
            f"[evaluate_model] Metric computation failed for {model_name}: {e}"
        ) from e

    print(f"\n{'='*50}")
    print(f"  Evaluation: {model_name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:12s}: {v}")

    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    ))
    return metrics


def plot_confusion_matrix(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    threshold: float = 0.5,
) -> None:
    """
    Plot confusion matrix with counts and percentages.
    Shows TP, FP, TN, FN with business context labels.
    """
    try:
        _validate_eval_inputs(model, X_test, y_test, context="plot_confusion_matrix")
        _validate_threshold(threshold)
    except (AttributeError, TypeError, ValueError) as e:
        raise type(e)(f"[plot_confusion_matrix] Validation failed: {e}") from e

    X_test = _clean_column_names(X_test)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
    except Exception as e:
        raise RuntimeError(
            f"[plot_confusion_matrix] Prediction failed for {model_name}: {e}"
        ) from e

    if cm.shape != (2, 2):
        raise ValueError(
            f"[plot_confusion_matrix] Expected 2x2 confusion matrix, "
            f"got {cm.shape}. Check that y_test has exactly 2 classes."
        )

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = np.array([
        [f"TN\n{cm[0,0]:,}\n(Correct — Legit)", f"FP\n{cm[0,1]:,}\n(False Alarm)"],
        [f"FN\n{cm[1,0]:,}\n(Missed Fraud)",     f"TP\n{cm[1,1]:,}\n(Caught Fraud)"]
    ])
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues",
                xticklabels=["Predicted Legit", "Predicted Fraud"],
                yticklabels=["Actual Legit", "Actual Fraud"],
                ax=ax, linewidths=1)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)
    plt.tight_layout()
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    total_fraud = tp + fn
    if total_fraud == 0:
        print("[plot_confusion_matrix] Warning: no actual fraud in test set")
        return

    print(f"\nBusiness interpretation:")
    print(f"  Fraud caught  : {tp:,} / {total_fraud:,} ({tp/total_fraud*100:.1f}%)")
    print(f"  Fraud missed  : {fn:,} / {total_fraud:,} ({fn/total_fraud*100:.1f}%) ← financial loss")
    legit_total = fp + tn
    if legit_total > 0:
        print(f"  False alarms  : {fp:,} ({fp/legit_total*100:.2f}% of legit) ← customer friction")


def plot_precision_recall_curve(
    models_dict: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    title: str = "Precision-Recall Curves"
) -> None:
    """
    Plot Precision-Recall curves for multiple models on one chart.
    AUC-PR shown in legend for each model.
    """
    if not models_dict:
        raise ValueError("[plot_precision_recall_curve] models_dict is empty")
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"[plot_precision_recall_curve] X_test must be DataFrame")

    X_test = _clean_column_names(X_test)
    colors = [BLUE, RED, "#2ECC71", "#F39C12", "#9B59B6"]
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (key, info) in enumerate(models_dict.items()):
        if "model" not in info:
            print(f"[plot_precision_recall_curve] Warning: no 'model' key for '{key}', skipping")
            continue
        try:
            y_prob = info["model"].predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            auc_pr = average_precision_score(y_test, y_prob)
            ax.plot(recall, precision,
                    color=colors[i % len(colors)],
                    linewidth=2,
                    label=f"{info['name']} (AUC-PR={auc_pr:.4f})")
        except Exception as e:
            print(f"[plot_precision_recall_curve] Warning: failed for '{key}': {e}")
            continue

    baseline = y_test.mean()
    ax.axhline(baseline, linestyle="--", color="gray",
               label=f"Random baseline ({baseline:.3f})")
    ax.set_xlabel("Recall (Fraud Detection Rate)", fontsize=11)
    ax.set_ylabel("Precision (Fraud Flag Accuracy)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.show()


def plot_roc_curves(
    models_dict: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    title: str = "ROC Curves"
) -> None:
    """Plot ROC curves for multiple models."""
    if not models_dict:
        raise ValueError("[plot_roc_curves] models_dict is empty")

    X_test = _clean_column_names(X_test)
    colors = [BLUE, RED, "#2ECC71", "#F39C12", "#9B59B6"]
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (key, info) in enumerate(models_dict.items()):
        if "model" not in info:
            print(f"[plot_roc_curves] Warning: no 'model' key for '{key}', skipping")
            continue
        try:
            y_prob = info["model"].predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, color=colors[i % len(colors)],
                    linewidth=2,
                    label=f"{info['name']} (AUC={auc:.4f})")
        except Exception as e:
            print(f"[plot_roc_curves] Warning: failed for '{key}': {e}")
            continue

    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# MODEL COMPARISON
# ─────────────────────────────────────────────

def compare_models(
    models_dict: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate all models and return a side-by-side comparison table.
    Sorted by AUC-PR descending.
    Models that fail evaluation are skipped with a warning.
    """
    if not models_dict:
        raise ValueError("[compare_models] models_dict is empty")

    rows = []
    for key, info in models_dict.items():
        if "model" not in info:
            print(f"[compare_models] Warning: no 'model' key for '{key}', skipping")
            continue
        try:
            metrics = evaluate_model(
                info["model"], X_test, y_test,
                model_name=info["name"]
            )
            rows.append(metrics)
        except Exception as e:
            print(f"[compare_models] Warning: evaluation failed for '{key}': {e}")
            continue

    if not rows:
        raise RuntimeError(
            "[compare_models] All model evaluations failed. "
            "Check that your models were trained on compatible data."
        )

    comparison_df = (pd.DataFrame(rows)
                     .sort_values("auc_pr", ascending=False)
                     .reset_index(drop=True))

    print(f"\n{'='*60}")
    print(f"  MODEL COMPARISON (sorted by AUC-PR)")
    print(f"{'='*60}")
    print(comparison_df.to_string(index=False))
    return comparison_df


def plot_model_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Bar chart comparing all models across AUC-PR, F1, Precision, Recall, ROC-AUC.
    """
    if comparison_df is None or len(comparison_df) == 0:
        raise ValueError("[plot_model_comparison] comparison_df is empty")

    required_cols = {"model", "auc_pr", "f1", "precision", "recall", "roc_auc"}
    missing = required_cols - set(comparison_df.columns)
    if missing:
        raise ValueError(
            f"[plot_model_comparison] Missing columns in comparison_df: {missing}"
        )

    metrics   = ["auc_pr", "f1", "precision", "recall", "roc_auc"]
    n_metrics = len(metrics)
    n_models  = len(comparison_df)
    colors    = ["#4C9BE8", "#E8524C", "#2ECC71", "#F39C12", "#9B59B6"]

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))

    for i, metric in enumerate(metrics):
        bars = axes[i].bar(
            comparison_df["model"],
            comparison_df[metric],
            color=colors[:n_models],
            edgecolor="white",
            alpha=0.85
        )
        for bar in bars:
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8
            )
        axes[i].set_title(metric.upper().replace("_", "-"), fontsize=11)
        axes[i].set_ylim(0, 1.1)
        axes[i].tick_params(axis="x", rotation=30)

    plt.suptitle("Model Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_cv_comparison(results_dict: Dict) -> None:
    """
    Plot cross-validation AUC-PR mean ± std for all models.
    Error bars show standard deviation across folds.
    """
    if not results_dict:
        raise ValueError("[plot_cv_comparison] results_dict is empty")

    names, means, stds = [], [], []
    for key, info in results_dict.items():
        cv = info.get("cv_results")
        if cv is None:
            print(f"[plot_cv_comparison] No CV results for '{key}', skipping")
            continue
        if "auc_pr" not in cv.columns:
            print(f"[plot_cv_comparison] No 'auc_pr' column in CV results for '{key}', skipping")
            continue
        names.append(info.get("name", key))
        means.append(cv["auc_pr"].mean())
        stds.append(cv["auc_pr"].std())

    if not names:
        print("[plot_cv_comparison] No valid CV results found — nothing to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C9BE8", "#E8524C", "#2ECC71", "#F39C12"]
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=colors[:len(names)], edgecolor="white", alpha=0.85)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.005,
            f"{mean:.4f}\n±{std:.4f}",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_ylabel("AUC-PR")
    ax.set_title("Cross-Validation AUC-PR — Mean ± Std (5-Fold)", fontsize=13)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.show()
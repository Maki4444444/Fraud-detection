"""
explainability.py
-----------------
Task 3 — Model explainability using SHAP and built-in feature importance.

Covers:
  - Built-in feature importance (tree-based models)
  - SHAP summary plots (global importance)
  - SHAP force plots (individual predictions)
  - SHAP waterfall plots (individual predictions)
  - Built-in vs SHAP comparison
  - Case finding helpers (TP, FP, FN)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[explainability] Warning: shap not installed. Run: pip install shap")

BLUE = "#4C9BE8"
RED  = "#E8524C"


# ─────────────────────────────────────────────
# INPUT VALIDATION HELPERS
# ─────────────────────────────────────────────

def _check_shap_available() -> None:
    if not SHAP_AVAILABLE:
        raise ImportError(
            "[explainability] shap is not installed. Run: pip install shap"
        )


def _validate_dataframe(X: pd.DataFrame, name: str = "X") -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(X).__name__}")
    if len(X) == 0:
        raise ValueError(f"{name} is empty — no rows")


def _validate_array(arr: np.ndarray, name: str = "array") -> None:
    if arr is None:
        raise ValueError(f"{name} is None — run compute_shap_values() first")
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(arr).__name__}")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D (n_samples x n_features), got shape {arr.shape}")


def _clean_column_names(X: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names — mirror of model_trainer._validate_column_names."""
    import re
    X = X.copy()
    X.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in X.columns]
    return X


# ─────────────────────────────────────────────
# BUILT-IN FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: List[str],
    model_name: str = "Model",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Plot and return built-in feature importance for tree-based models.
    Works with Random Forest, XGBoost, and LightGBM.

    Why built-in importance can be misleading:
      - Random Forest: mean decrease in impurity — biased toward
        high-cardinality features
      - XGBoost/LightGBM: gain-based importance is more reliable
      - SHAP provides the most accurate importance (section 2 below)
    """
    if not feature_names:
        raise ValueError("[plot_feature_importance] feature_names is empty")
    if top_n < 1:
        raise ValueError(f"[plot_feature_importance] top_n must be >= 1, got {top_n}")
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"[plot_feature_importance] Model {type(model).__name__} does not have "
            f"feature_importances_. Only tree-based models (RF, XGBoost, LightGBM) "
            f"support built-in importance. Use SHAP for Logistic Regression."
        )

    importances = model.feature_importances_

    if len(importances) != len(feature_names):
        raise ValueError(
            f"[plot_feature_importance] Length mismatch: "
            f"{len(importances)} importances vs {len(feature_names)} feature names. "
            f"Ensure feature_names matches the columns used during training."
        )

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        importance_df["feature"][::-1],
        importance_df["importance"][::-1],
        color=RED, alpha=0.8, edgecolor="white"
    )
    for bar in bars:
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}",
                va="center", fontsize=8)

    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(f"Top {top_n} Features — {model_name} Built-in Importance", fontsize=12)
    plt.tight_layout()
    plt.show()

    print(f"\nTop {top_n} features by built-in importance:")
    print(importance_df.to_string(index=False))
    return importance_df


# ─────────────────────────────────────────────
# SHAP ANALYSIS
# ─────────────────────────────────────────────

def get_shap_explainer(model, X_train: pd.DataFrame):
    """
    Create a SHAP TreeExplainer for tree-based models.

    Why TreeExplainer:
      Specifically designed for tree-based models (XGBoost, LightGBM,
      Random Forest). It is exact (not approximate) and very fast.
      For Logistic Regression use shap.LinearExplainer instead.
    """
    _check_shap_available()
    _validate_dataframe(X_train, name="X_train")

    X_train = _clean_column_names(X_train)

    if not hasattr(model, "predict"):
        raise AttributeError(
            f"[get_shap_explainer] Model {type(model).__name__} does not appear to be fitted."
        )

    try:
        print("[get_shap_explainer] Creating TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        print("[get_shap_explainer] Done.")
        return explainer
    except Exception as e:
        raise RuntimeError(
            f"[get_shap_explainer] Failed to create TreeExplainer for "
            f"{type(model).__name__}.\n"
            f"  If using Logistic Regression, use shap.LinearExplainer instead.\n"
            f"  Error: {e}"
        ) from e


def compute_shap_values(
    explainer,
    X: pd.DataFrame,
    sample_size: int = 1000,
    random_state: int = 42,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values for a sample of the data.

    Why sample:
      Computing SHAP values for all rows is slow on large datasets.
      A sample of 1000–2000 gives a reliable global picture.
      Use 5000+ for final reports.

    Returns
    -------
    shap_values : 2D array (n_samples x n_features)
    X_sample    : the sampled DataFrame
    """
    _check_shap_available()
    _validate_dataframe(X, name="X")

    if sample_size < 1:
        raise ValueError(
            f"[compute_shap_values] sample_size must be >= 1, got {sample_size}"
        )

    X = _clean_column_names(X)

    if sample_size < len(X):
        X_sample = X.sample(sample_size, random_state=random_state)
        print(f"[compute_shap_values] Using sample of {sample_size:,} rows "
              f"(full size: {len(X):,})")
    else:
        X_sample = X.copy()
        print(f"[compute_shap_values] Using all {len(X):,} rows")

    print("[compute_shap_values] Computing SHAP values (this may take a minute)...")
    try:
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        raise RuntimeError(
            f"[compute_shap_values] SHAP computation failed.\n"
            f"  Common causes: column mismatch between training and this data, "
            f"or model was not fitted.\n"
            f"  Error: {e}"
        ) from e

    # Binary classifiers sometimes return list [class0_vals, class1_vals]
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values = shap_values[1]  # take fraud class
        else:
            raise ValueError(
                f"[compute_shap_values] Unexpected SHAP values format: "
                f"list of length {len(shap_values)}"
            )

    if shap_values.ndim != 2:
        raise ValueError(
            f"[compute_shap_values] Expected 2D SHAP values array, "
            f"got shape {shap_values.shape}"
        )

    print(f"[compute_shap_values] Done. Shape: {shap_values.shape}")
    return shap_values, X_sample


# ─────────────────────────────────────────────
# SHAP PLOTS
# ─────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    model_name: str = "Model",
    max_display: int = 15,
) -> None:
    """
    SHAP summary plot — global feature importance with direction.

    How to read:
      Y-axis: features ranked by mean |SHAP value| (most important at top)
      X-axis: SHAP value (positive = pushes toward fraud prediction)
      Color : feature value (red = high, blue = low)
    """
    _check_shap_available()
    _validate_array(shap_values, name="shap_values")
    _validate_dataframe(X_sample, name="X_sample")

    if shap_values.shape[0] != len(X_sample):
        raise ValueError(
            f"[plot_shap_summary] Shape mismatch: "
            f"shap_values has {shap_values.shape[0]} rows "
            f"but X_sample has {len(X_sample)} rows"
        )
    if shap_values.shape[1] != X_sample.shape[1]:
        raise ValueError(
            f"[plot_shap_summary] Feature count mismatch: "
            f"shap_values has {shap_values.shape[1]} features "
            f"but X_sample has {X_sample.shape[1]} columns"
        )
    if max_display < 1:
        raise ValueError(f"[plot_shap_summary] max_display must be >= 1")

    try:
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sample,
                          max_display=max_display, show=False)
        plt.title(f"SHAP Summary Plot — {model_name}", fontsize=13, pad=20)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        raise RuntimeError(
            f"[plot_shap_summary] Failed to generate summary plot: {e}"
        ) from e


def plot_shap_bar(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    model_name: str = "Model",
    max_display: int = 15,
) -> pd.DataFrame:
    """
    SHAP bar chart — mean absolute SHAP value per feature.
    Cleaner comparison against built-in feature importance.
    Returns a DataFrame of SHAP importances.
    """
    _check_shap_available()
    _validate_array(shap_values, name="shap_values")
    _validate_dataframe(X_sample, name="X_sample")

    if shap_values.shape[1] != X_sample.shape[1]:
        raise ValueError(
            f"[plot_shap_bar] Feature count mismatch: "
            f"{shap_values.shape[1]} vs {X_sample.shape[1]}"
        )

    mean_shap = pd.DataFrame({
        "feature":          X_sample.columns,
        "shap_importance":  np.abs(shap_values).mean(axis=0)
    }).sort_values("shap_importance", ascending=False).head(max_display)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(mean_shap["feature"][::-1],
            mean_shap["shap_importance"][::-1],
            color=RED, alpha=0.8, edgecolor="white")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"SHAP Feature Importance — {model_name}", fontsize=12)
    plt.tight_layout()
    plt.show()

    return mean_shap


def plot_shap_force(
    explainer,
    X_sample: pd.DataFrame,
    idx: int,
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> None:
    """
    SHAP force plot for a single prediction.
    Shows which features pushed the prediction toward or away from fraud.
    """
    _check_shap_available()
    _validate_dataframe(X_sample, name="X_sample")

    if idx < 0 or idx >= len(X_sample):
        raise IndexError(
            f"[plot_shap_force] idx={idx} is out of bounds for "
            f"X_sample with {len(X_sample)} rows. "
            f"Valid range: 0 to {len(X_sample)-1}"
        )
    if len(y_true) != len(X_sample):
        raise ValueError(
            f"[plot_shap_force] y_true length ({len(y_true)}) != "
            f"X_sample length ({len(X_sample)})"
        )
    if len(y_pred) != len(X_sample):
        raise ValueError(
            f"[plot_shap_force] y_pred length ({len(y_pred)}) != "
            f"X_sample length ({len(X_sample)})"
        )

    true_label = int(y_true.iloc[idx])
    pred_label = int(y_pred[idx])
    label_map  = {
        (1, 1): "True Positive  (Caught Fraud)",
        (0, 0): "True Negative  (Correct Legit)",
        (1, 0): "False Negative (Missed Fraud)",
        (0, 1): "False Positive (False Alarm)"
    }
    case_label = label_map.get((true_label, pred_label), "Unknown")

    print(f"\n[force_plot] Index: {idx} | True: {true_label} | Predicted: {pred_label}")
    print(f"[force_plot] Case : {case_label}")
    print(f"[force_plot] Feature values:")
    print(X_sample.iloc[idx].to_string())

    try:
        shap_vals = explainer.shap_values(X_sample.iloc[[idx]])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        expected_value = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )

        shap.initjs()
        shap.force_plot(
            expected_value,
            shap_vals[0],
            X_sample.iloc[idx],
            matplotlib=True,
            show=False
        )
        plt.title(f"{model_name} — {case_label}", fontsize=11, pad=30)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        raise RuntimeError(
            f"[plot_shap_force] Force plot failed at index {idx}: {e}"
        ) from e


def plot_shap_waterfall(
    explainer,
    X_sample: pd.DataFrame,
    idx: int,
    model_name: str = "Model",
) -> None:
    """
    SHAP waterfall plot for a single prediction.
    Shows step-by-step how each feature contributed to the final score.
    Often clearer than force plots for individual case analysis.
    """
    _check_shap_available()
    _validate_dataframe(X_sample, name="X_sample")

    if idx < 0 or idx >= len(X_sample):
        raise IndexError(
            f"[plot_shap_waterfall] idx={idx} out of bounds for "
            f"X_sample with {len(X_sample)} rows"
        )

    try:
        shap_vals = explainer(X_sample.iloc[[idx]])

        # Handle multi-output (binary classifier returns 3D array)
        if hasattr(shap_vals, "values") and shap_vals.values.ndim == 3:
            shap_vals = shap.Explanation(
                values=shap_vals.values[0, :, 1],
                base_values=shap_vals.base_values[0, 1],
                data=shap_vals.data[0],
                feature_names=X_sample.columns.tolist()
            )
        else:
            shap_vals = shap_vals[0]

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_vals, show=False)
        plt.title(f"{model_name} — Waterfall Plot (index {idx})", fontsize=11)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        raise RuntimeError(
            f"[plot_shap_waterfall] Waterfall plot failed at index {idx}: {e}"
        ) from e


# ─────────────────────────────────────────────
# COMPARISON: SHAP vs BUILT-IN
# ─────────────────────────────────────────────

def compare_importances(
    builtin_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Side-by-side comparison of built-in vs SHAP feature importance.

    Differences reveal:
      - Features important by impurity but not by SHAP → likely spurious
      - Features ranked higher by SHAP than built-in → true contribution
        was underestimated by tree splits
    """
    if builtin_df is None or len(builtin_df) == 0:
        raise ValueError("[compare_importances] builtin_df is empty or None")
    if shap_df is None or len(shap_df) == 0:
        raise ValueError("[compare_importances] shap_df is empty or None")
    if "feature" not in builtin_df.columns or "importance" not in builtin_df.columns:
        raise ValueError(
            "[compare_importances] builtin_df must have 'feature' and 'importance' columns"
        )
    if "feature" not in shap_df.columns or "shap_importance" not in shap_df.columns:
        raise ValueError(
            "[compare_importances] shap_df must have 'feature' and 'shap_importance' columns"
        )

    builtin_top = builtin_df.head(top_n)[["feature", "importance"]].copy()
    builtin_top["builtin_rank"] = range(1, len(builtin_top) + 1)

    shap_top = shap_df.head(top_n)[["feature", "shap_importance"]].copy()
    shap_top["shap_rank"] = range(1, len(shap_top) + 1)

    merged = pd.merge(builtin_top, shap_top, on="feature", how="outer")
    merged = merged.sort_values("shap_rank")

    print("\nBuilt-in Importance vs SHAP Importance:")
    print(merged.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    top_builtin = builtin_df.head(top_n)
    axes[0].barh(top_builtin["feature"][::-1],
                 top_builtin["importance"][::-1],
                 color=BLUE, alpha=0.8, edgecolor="white")
    axes[0].set_title("Built-in Feature Importance", fontsize=11)
    axes[0].set_xlabel("Importance Score")

    top_shap = shap_df.head(top_n)
    axes[1].barh(top_shap["feature"][::-1],
                 top_shap["shap_importance"][::-1],
                 color=RED, alpha=0.8, edgecolor="white")
    axes[1].set_title("SHAP Feature Importance", fontsize=11)
    axes[1].set_xlabel("Mean |SHAP Value|")

    plt.suptitle("Built-in vs SHAP Feature Importance", fontsize=13)
    plt.tight_layout()
    plt.show()

    return merged


# ─────────────────────────────────────────────
# CASE FINDING HELPERS
# ─────────────────────────────────────────────

def find_cases(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n: int = 1,
) -> Dict[str, List[int]]:
    """
    Find indices for true positives, false positives, and false negatives.
    Selects the most confident predictions for the most informative force plots.

    Returns
    -------
    dict with keys: 'tp', 'fp', 'fn'
    Each value: list of n row indices
    """
    if not isinstance(y_true, pd.Series):
        raise TypeError(
            f"[find_cases] y_true must be pandas Series, got {type(y_true).__name__}"
        )
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(
            f"[find_cases] y_pred must be numpy array, got {type(y_pred).__name__}"
        )
    if not isinstance(y_prob, np.ndarray):
        raise TypeError(
            f"[find_cases] y_prob must be numpy array, got {type(y_prob).__name__}"
        )
    if len(y_true) != len(y_pred) or len(y_true) != len(y_prob):
        raise ValueError(
            f"[find_cases] Length mismatch: "
            f"y_true={len(y_true)}, y_pred={len(y_pred)}, y_prob={len(y_prob)}"
        )
    if n < 1:
        raise ValueError(f"[find_cases] n must be >= 1, got {n}")

    y_true_arr = np.array(y_true)

    tp_idx = np.where((y_true_arr == 1) & (y_pred == 1))[0]
    fp_idx = np.where((y_true_arr == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true_arr == 1) & (y_pred == 0))[0]

    # Sort by confidence: highest prob for TP/FP, lowest prob for FN
    tp_sorted = tp_idx[np.argsort(y_prob[tp_idx])[::-1]][:n] if len(tp_idx) > 0 else []
    fp_sorted = fp_idx[np.argsort(y_prob[fp_idx])[::-1]][:n] if len(fp_idx) > 0 else []
    fn_sorted = fn_idx[np.argsort(y_prob[fn_idx])][:n]       if len(fn_idx) > 0 else []

    print(f"\n[find_cases] Found:")
    print(f"  True Positives  (TP): {len(tp_idx):,} total | showing {len(tp_sorted)}")
    print(f"  False Positives (FP): {len(fp_idx):,} total | showing {len(fp_sorted)}")
    print(f"  False Negatives (FN): {len(fn_idx):,} total | showing {len(fn_sorted)}")

    if len(tp_idx) == 0:
        print("  Warning: no True Positives found — model may have threshold issues")
    if len(fp_idx) == 0:
        print("  Warning: no False Positives found — very high precision model")
    if len(fn_idx) == 0:
        print("  Warning: no False Negatives found — perfect recall on this sample")

    return {
        "tp": list(tp_sorted),
        "fp": list(fp_sorted),
        "fn": list(fn_sorted)
    }
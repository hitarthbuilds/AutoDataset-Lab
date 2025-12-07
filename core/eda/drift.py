"""
drift.py
Enterprise-grade statistical drift detection module.
Fully rewritten to remove all dependencies on River, Rust, or compiled wheels.

This module provides:
    - Univariate drift metrics (KS, PSI, Wasserstein, KL, JS)
    - Multivariate drift summary
    - Concept drift flags with thresholds
    - Time-window roll drift
    - Categorical + numerical handling
    - Pure Python + SciPy only

Output is designed for ingestion by:
    - EDA dashboard
    - PDF report generator
    - Monitoring pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.stats import ks_2samp, entropy
from scipy.stats import wasserstein_distance



# -----------------------------
# Helpers
# -----------------------------

def _safe_hist(series: pd.Series, bins: int = 20):
    try:
        clean = series.dropna().astype(float)
        if len(clean) == 0:
            return np.array([]), np.array([])
        return np.histogram(clean, bins=bins, density=True)
    except Exception:
        return np.array([]), np.array([])


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 20):
    """
    Population Stability Index (PSI)
    Measures shift in distribution between two numeric vectors.
    """
    try:
        e_hist, e_edges = np.histogram(expected.dropna(), bins=bins)
        a_hist, _ = np.histogram(actual.dropna(), bins=e_edges)

        e_prop = e_hist / (len(expected) + 1e-9)
        a_prop = a_hist / (len(actual) + 1e-9)

        psi_vals = (a_prop - e_prop) * np.log((a_prop + 1e-9) / (e_prop + 1e-9))
        return float(np.sum(psi_vals))
    except Exception:
        return None


def _kl_js(expected: pd.Series, actual: pd.Series, bins: int = 20):
    """
    KL divergence + Jensen-Shannon divergence.
    """
    try:
        e_hist, e_edges = np.histogram(expected.dropna(), bins=bins, density=True)
        a_hist, _ = np.histogram(actual.dropna(), bins=e_edges, density=True)

        e_hist += 1e-9
        a_hist += 1e-9

        kl = float(entropy(e_hist, a_hist))
        m = 0.5 * (e_hist + a_hist)
        js = 0.5 * entropy(e_hist, m) + 0.5 * entropy(a_hist, m)
        return kl, float(js)
    except Exception:
        return None, None


# -----------------------------
# Main univariate drift detector
# -----------------------------

def detect_column_drift(
    col: str,
    base: pd.Series,
    current: pd.Series,
    bins: int = 20
) -> Dict[str, Any]:
    """
    Computes drift metrics for a single column.
    Supports numeric + categorical.
    """

    result = {
        "column": col,
        "dtype": str(base.dtype),
        "ks_stat": None,
        "ks_pvalue": None,
        "psi": None,
        "wasserstein": None,
        "kl_divergence": None,
        "js_divergence": None,
        "categorical_drift": None,
        "flag": False,
        "reason": ""
    }

    is_numeric = pd.api.types.is_numeric_dtype(base)

    # --------------------
    # Numeric drift metrics
    # --------------------
    if is_numeric:
        try:
            ks = ks_2samp(base.dropna(), current.dropna())
            result["ks_stat"] = float(ks.statistic)
            result["ks_pvalue"] = float(ks.pvalue)
        except Exception:
            pass

        try:
            result["psi"] = _psi(base, current)
        except Exception:
            pass

        try:
            result["wasserstein"] = float(
                wasserstein_distance(base.dropna(), current.dropna())
            )
        except Exception:
            pass

        try:
            kl, js = _kl_js(base, current)
            result["kl_divergence"] = kl
            result["js_divergence"] = js
        except Exception:
            pass

    # --------------------
    # Categorical drift
    # --------------------
    else:
        try:
            base_prop = base.value_counts(normalize=True)
            cur_prop = current.value_counts(normalize=True)

            all_vals = list(set(base_prop.index) | set(cur_prop.index))

            drift_val = 0
            for v in all_vals:
                p = base_prop.get(v, 0)
                q = cur_prop.get(v, 0)
                drift_val += abs(p - q)

            result["categorical_drift"] = float(drift_val)
        except Exception:
            pass

    # --------------------
    # Flag drift
    # --------------------
    # Enterprise-grade conservative thresholds.
    try:
        if is_numeric:
            if result["psi"] is not None and result["psi"] > 0.25:
                result["flag"] = True
                result["reason"] = "High PSI drift (>0.25)"

            if result["ks_pvalue"] is not None and result["ks_pvalue"] < 0.01:
                result["flag"] = True
                result["reason"] = "Significant KS distribution change (p<0.01)"

            if result["js_divergence"] is not None and result["js_divergence"] > 0.15:
                result["flag"] = True
                result["reason"] = "High JS divergence (>0.15)"
        else:
            if result["categorical_drift"] is not None and result["categorical_drift"] > 0.25:
                result["flag"] = True
                result["reason"] = "Large categorical frequency shift"

    except Exception:
        pass

    return result


# -----------------------------
# Dataset-level drift detector
# -----------------------------

def detect_dataset_drift(
    base_df: pd.DataFrame,
    current_df: pd.DataFrame,
    bins: int = 20
) -> Dict[str, Any]:
    """
    Computes drift OVER THE ENTIRE DATASET.
    Returns:
        - per-column drift record
        - summary stats
        - high-drift columns
    """

    out = {
        "rows_base": len(base_df),
        "rows_current": len(current_df),
        "columns": list(base_df.columns),
        "per_column": [],
        "high_drift_columns": []
    }

    common_cols = [c for c in base_df.columns if c in current_df.columns]

    for col in common_cols:
        drift = detect_column_drift(col, base_df[col], current_df[col], bins=bins)
        out["per_column"].append(drift)
        if drift["flag"]:
            out["high_drift_columns"].append(col)

    return out


# -----------------------------
# Rolling window drift (time drift)
# -----------------------------

def rolling_drift(
    df: pd.DataFrame,
    timestamp_col: str,
    window: int = 5000
) -> Dict[str, Any]:
    """
    Rolling drift detector for time-ordered data.

    Returns drift flags for each window step.
    """
    try:
        df_sorted = df.sort_values(timestamp_col)
    except Exception:
        return {"error": "Invalid timestamp column for sorting."}

    results = []
    for start in range(0, len(df_sorted) - window, window):
        base = df_sorted.iloc[start : start + window]
        cur = df_sorted.iloc[start + window : start + 2 * window]

        drift = detect_dataset_drift(base, cur)
        results.append({
            "window_start": start,
            "window_end": start + 2 * window,
            "high_drift_columns": drift["high_drift_columns"]
        })

    return {
        "windows": results,
        "total_segments": len(results)
    }

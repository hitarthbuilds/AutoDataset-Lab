# core/eda/anomalies.py
"""
Enterprise-grade anomaly detection module for AutoDataset-Lab.

Detectors implemented:
- Isolation Forest (global row-level)
- Isolation Forest (per-column)
- Local Outlier Factor (LOF)
- Z-score outliers
- IQR outliers

Returns:
- Fully JSON-serializable dictionary
- Per-column anomaly counts
- Row-level anomaly flags
- Detector comparison summary
- Useful metadata for UI and PDF reports
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# ------------------------------ Helpers ------------------------------

def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dataframe to numeric where possible, non-numeric → NaN."""
    out = {}
    for c in df.columns:
        try:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            out[c] = pd.Series([np.nan] * len(df), index=df.index)
    return pd.DataFrame(out)


def _safe_list(x):
    """Ensure JSON convertibility."""
    try:
        return list(map(lambda v: float(v) if isinstance(v, (int, float, np.floating)) else v, x))
    except Exception:
        return []


def _limit_idx(lst, n=50):
    """Return only first n indexes."""
    try:
        return lst[:n]
    except Exception:
        return []


# ------------------------------ Z-score ------------------------------

def _zscore_detector(df_num: pd.DataFrame, threshold: float = 3.5) -> Dict[str, Any]:
    """Column-wise Z-score anomalies."""
    out = {}
    try:
        z = (df_num - df_num.mean()) / df_num.std(ddof=0)
        z_abs = z.abs()
        for col in df_num.columns:
            try:
                mask = z_abs[col] > threshold
                out[col] = int(mask.sum())
            except Exception:
                out[col] = 0
    except Exception:
        pass
    return {"method": "zscore", "per_column": out}


# ------------------------------ IQR ------------------------------

def _iqr_detector(df_num: pd.DataFrame, k: float = 1.5) -> Dict[str, Any]:
    """Column-wise IQR anomalies."""
    out = {}
    try:
        for col in df_num.columns:
            ser = df_num[col].dropna()
            if ser.empty:
                out[col] = 0
                continue

            q1 = np.percentile(ser, 25)
            q3 = np.percentile(ser, 75)
            iqr = q3 - q1

            lower = q1 - k * iqr
            upper = q3 + k * iqr

            mask = (df_num[col] < lower) | (df_num[col] > upper)
            out[col] = int(mask.sum())
    except Exception:
        pass

    return {"method": "iqr", "per_column": out}


# ------------------------------ Isolation Forest ------------------------------

def _isolation_forest_global(df_num: pd.DataFrame) -> Dict[str, Any]:
    """
    Isolation Forest row-level anomaly detection.
    Returns: anomaly row indexes, count.
    """
    try:
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=1)
        preds = iso.fit_predict(df_num.fillna(0))  # -1 = anomaly
        mask = preds == -1
        idx = df_num.index[mask].tolist()
        return {
            "method": "isolation_forest_global",
            "anomaly_count": int(mask.sum()),
            "anomaly_index_sample": _limit_idx(idx, 50)
        }
    except Exception:
        return {
            "method": "isolation_forest_global",
            "anomaly_count": None,
            "anomaly_index_sample": []
        }


def _isolation_forest_per_column(df_num: pd.DataFrame) -> Dict[str, Any]:
    """
    Per-column isolation forest (1 feature at a time).
    More expensive but interpretable.
    """
    out = {}
    try:
        for col in df_num.columns:
            try:
                ser = df_num[[col]].fillna(0)
                iso = IsolationForest(n_estimators=160, contamination="auto", random_state=1)
                preds = iso.fit_predict(ser)
                out[col] = int((preds == -1).sum())
            except Exception:
                out[col] = 0
    except Exception:
        pass
    return {"method": "isolation_forest_column", "per_column": out}


# ------------------------------ Local Outlier Factor ------------------------------

def _lof_detector(df_num: pd.DataFrame, n_neighbors: int = 20) -> Dict[str, Any]:
    """
    LOF: Local Outlier Factor.
    """
    try:
        lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(df_num)-1), contamination="auto")
        preds = lof.fit_predict(df_num.fillna(0))
        mask = preds == -1
        idx = df_num.index[mask].tolist()
        return {
            "method": "lof",
            "anomaly_count": int(mask.sum()),
            "anomaly_index_sample": _limit_idx(idx, 50)
        }
    except Exception:
        return {"method": "lof", "anomaly_count": None, "anomaly_index_sample": []}


# ------------------------------ Master API ------------------------------

def detect_anomalies(
    df: pd.DataFrame,
    max_rows: int = 30000
) -> Dict[str, Any]:
    """
    Main anomaly detection pipeline.

    Returns JSON-safe dictionary containing:
    - methods: {method_name: {...}}
    - per_column_combined: anomaly counts aggregated
    - global_summary
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    # Sample for large datasets to avoid meltdown
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=1)

    # Convert to numeric-friendly df
    df_num = _safe_numeric(df)
    numeric_cols = df_num.columns.tolist()

    # Run detectors
    results = {}

    # Z-score
    results["zscore"] = _zscore_detector(df_num)

    # IQR
    results["iqr"] = _iqr_detector(df_num)

    # Isolation Forest global
    results["isolation_forest_global"] = _isolation_forest_global(df_num)

    # Isolation Forest per-column
    results["isolation_forest_column"] = _isolation_forest_per_column(df_num)

    # LOF
    results["lof"] = _lof_detector(df_num)

    # Aggregate per-column anomaly counts
    per_column_total = {col: 0 for col in numeric_cols}
    for mname, mres in results.items():
        if "per_column" in mres:
            for col, val in mres["per_column"].items():
                try:
                    per_column_total[col] += int(val or 0)
                except Exception:
                    pass

    # Global summary
    total_global = 0
    for name in results:
        if "anomaly_count" in results[name]:
            try:
                total_global += int(results[name]["anomaly_count"] or 0)
            except Exception:
                pass

    summary = {
        "total_global_anomalies": int(total_global),
        "most_anomalous_columns": sorted(
            per_column_total.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
    }

    return {
        "methods": results,
        "per_column_combined": per_column_total,
        "summary": summary,
        "meta": {
            "rows_used": int(len(df)),
            "numeric_columns_analyzed": numeric_cols,
        }
    }

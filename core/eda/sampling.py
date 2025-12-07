"""
sampling.py
ENTERPRISE-GRADE SAMPLING ENGINE
--------------------------------------------------------------

Provides:
    - simple random sampling
    - stratified sampling (classification-aware)
    - time-series window sampling
    - cluster-based sampling (KMeans-lite fallback)
    - anomaly-boosted sampling (uses anomaly scores)
    - rare-class oversampling
    - summary metadata for reproducibility

All outputs are JSON-ready and Streamlit-ready.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

# -------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------

def _normalize_frac(frac: float) -> float:
    if frac <= 0:
        return 0.05
    if frac > 1:
        return 1.0
    return frac


def _safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().reset_index(drop=True)


# -------------------------------------------------------------
# 1. RANDOM SAMPLING
# -------------------------------------------------------------
def simple_sample(df: pd.DataFrame, frac: float = 0.1, seed: int = 42) -> Dict[str, Any]:
    frac = _normalize_frac(frac)
    n = len(df)
    sample = df.sample(frac=frac, random_state=seed)
    return {
        "method": "simple_random",
        "fraction": frac,
        "input_rows": n,
        "sample_rows": len(sample),
        "sample_df": sample
    }


# -------------------------------------------------------------
# 2. STRATIFIED SAMPLING
# -------------------------------------------------------------
def stratified_sample(df: pd.DataFrame, target_col: str, frac: float = 0.1, seed: int = 42):
    if target_col not in df.columns:
        return {"error": f"{target_col} not found"}

    frac = _normalize_frac(frac)
    n = len(df)

    try:
        sample = df.groupby(target_col, group_keys=False).apply(
            lambda x: x.sample(frac=frac, random_state=seed)
        )
    except Exception:
        # fallback to basic
        sample = df.sample(frac=frac, random_state=seed)

    return {
        "method": "stratified",
        "target": target_col,
        "fraction": frac,
        "class_distribution_input": df[target_col].value_counts(normalize=True).to_dict(),
        "class_distribution_sample": sample[target_col].value_counts(normalize=True).to_dict(),
        "input_rows": n,
        "sample_rows": len(sample),
        "sample_df": _safe_copy(sample),
    }


# -------------------------------------------------------------
# 3. TIME SERIES SAMPLING
# -------------------------------------------------------------
def timeseries_sample(df: pd.DataFrame, datetime_col: str, window: str = "30D"):
    """
    Extracts a sliding window sample for time-series datasets.
    window: pandas offset, e.g., '7D', '30D', '3M', '1Y'
    """
    if datetime_col not in df.columns:
        return {"error": f"{datetime_col} not found"}

    try:
        df2 = df.copy()
        df2[datetime_col] = pd.to_datetime(df2[datetime_col], errors="coerce")
        df2 = df2.dropna(subset=[datetime_col]).sort_values(datetime_col)

        start = df2[datetime_col].min()
        end = start + pd.Timedelta(window)

        mask = (df2[datetime_col] >= start) & (df2[datetime_col] <= end)
        sample = df2.loc[mask]
    except Exception as e:
        return {"error": str(e)}

    return {
        "method": "timeseries_window",
        "datetime_col": datetime_col,
        "window": window,
        "window_start": str(start),
        "window_end": str(end),
        "input_rows": len(df),
        "sample_rows": len(sample),
        "sample_df": _safe_copy(sample),
    }


# -------------------------------------------------------------
# 4. CLUSTER SAMPLING (KMeans-lite approximation)
# -------------------------------------------------------------
def cluster_sample(df: pd.DataFrame, k: int = 5, per_cluster: int = 50, seed: int = 42):
    """
    Performs very lightweight KMeans-style sampling without sklearn KMeans.
    Uses random centroid initialization + iterative assignment (1 step).
    """

    if df.empty:
        return {"error": "empty dataset"}

    df_num = df.select_dtypes(include=["number"])
    if df_num.empty:
        return {"error": "no numeric columns for clustering"}

    np.random.seed(seed)
    X = df_num.to_numpy()

    # pick initial centroids
    try:
        idx = np.random.choice(len(X), size=min(k, len(X)), replace=False)
        centroids = X[idx]

        # assign clusters
        dist = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dist, axis=1)

        sample_idx = []
        for c in range(len(centroids)):
            members = np.where(labels == c)[0]
            if len(members) > 0:
                sel = np.random.choice(
                    members,
                    size=min(per_cluster, len(members)),
                    replace=False
                )
                sample_idx.extend(sel)

        sample = df.iloc[sample_idx]
    except Exception:
        # fallback random sample
        sample = df.sample(n=min(200, len(df)), random_state=seed)

    return {
        "method": "cluster_sample",
        "clusters": k,
        "input_rows": len(df),
        "sample_rows": len(sample),
        "sample_df": _safe_copy(sample),
    }


# -------------------------------------------------------------
# 5. ANOMALY BOOSTED SAMPLING
# -------------------------------------------------------------
def anomaly_boosted_sample(df: pd.DataFrame, anomalies: Dict[str, Any], pct_anomalies: float = 0.3, seed: int = 42):
    """
    Ensures anomaly-heavy regions are sampled.
    pct_anomalies = percent of sample dedicated to anomalous rows.
    """
    if not anomalies or "methods" not in anomalies:
        return {"error": "invalid anomalies dict, run anomaly detection first"}

    np.random.seed(seed)
    df2 = df.copy()

    # isolation forest anomaly labels if available
    iso_cols = anomalies.get("methods", {}).get("isolation_forest", {})
    if "columns" not in iso_cols:
        # fallback to random
        return simple_sample(df2, frac=0.2)

    # approximate anomaly density using per-column counts
    anomaly_score = np.zeros(len(df2))
    for col in df2.select_dtypes(include=["number"]).columns:
        try:
            col_vals = df2[col]
            z = (col_vals - col_vals.mean()) / (col_vals.std() + 1e-9)
            anomaly_score += np.abs(z)
        except Exception:
            pass

    # normalize
    anomaly_score = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min() + 1e-9)
    df2["_anomaly_score"] = anomaly_score

    # sample top anomalies
    cut = np.quantile(anomaly_score, 0.95)
    anomalous = df2[df2["_anomaly_score"] >= cut]
    normal = df2[df2["_anomaly_score"] < cut]

    n_total = min(500, len(df2))
    n_anom = int(n_total * pct_anomalies)
    n_norm = n_total - n_anom

    sample = pd.concat([
        anomalous.sample(n=min(n_anom, len(anomalous)), random_state=seed),
        normal.sample(n=min(n_norm, len(normal)), random_state=seed)
    ])

    sample = sample.drop(columns=["_anomaly_score"], errors="ignore")

    return {
        "method": "anomaly_boosted",
        "pct_anomalies": pct_anomalies,
        "input_rows": len(df),
        "sample_rows": len(sample),
        "sample_df": _safe_copy(sample),
    }


# -------------------------------------------------------------
# 6. RARE CLASS OVERSAMPLING
# -------------------------------------------------------------
def oversample_rare_class(df: pd.DataFrame, target_col: str, threshold: float = 0.05, seed: int = 42):
    """
    Boosts rare classes until each class >= threshold.
    """
    if target_col not in df.columns:
        return {"error": f"{target_col} not found"}

    y = df[target_col]
    dist = y.value_counts(normalize=True)

    rare = dist[dist < threshold].index.tolist()
    if not rare:
        return {"message": "no rare classes detected", "sample_df": df}

    np.random.seed(seed)
    df2 = df.copy()

    pieces = [df2]
    for cls in rare:
        subset = df2[df2[target_col] == cls]
        needed = int((threshold * len(df2)) - len(subset))
        if needed > 0:
            boost = subset.sample(n=min(len(subset), needed), replace=True, random_state=seed)
            pieces.append(boost)

    out = pd.concat(pieces).reset_index(drop=True)

    return {
        "method": "oversample_rare_classes",
        "rare_classes": rare,
        "input_rows": len(df),
        "output_rows": len(out),
        "sample_df": out,
    }

"""
utils.py
Shared helper utilities for the EDA engine.
These functions are intentionally lightweight and dependency-agnostic so they
can be used across all submodules (missing, drift, anomalies, quality, etc.)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import traceback
from typing import Tuple, Optional, Dict, List


# ------------------------------------------------------------
# Safe operations
# ------------------------------------------------------------

def safe_execute(fn, *args, **kwargs):
    """
    Executes a function safely and returns (result, error).
    Useful for EDA functions where failure should not stop the pipeline.
    """
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return None, traceback.format_exc()


# ------------------------------------------------------------
# Column helpers
# ------------------------------------------------------------

def split_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Splits columns into numeric, categorical, datetime, boolean.
    """
    numerics = df.select_dtypes(include=["number"]).columns.tolist()
    categoricals = df.select_dtypes(include=["object", "category"]).columns.tolist()
    booleans = df.select_dtypes(include=["bool"]).columns.tolist()
    datetimes = df.select_dtypes(include=["datetime64", "datetime"]).columns.tolist()

    return {
        "numeric": numerics,
        "categorical": categoricals,
        "boolean": booleans,
        "datetime": datetimes
    }


def memory_usage(df: pd.DataFrame) -> float:
    """
    Returns memory usage of DF in MB.
    """
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


# ------------------------------------------------------------
# Sampling utilities
# ------------------------------------------------------------

def sample_df(df: pd.DataFrame, max_rows: int = 50000) -> pd.DataFrame:
    """
    Downsamples massive datasets for faster EDA + visualizations.
    Smart sampling:
      - If rows <= max_rows -> return original
      - Else -> stratified sampling if a column called 'target' exists
    """
    if df.shape[0] <= max_rows:
        return df

    if "target" in df.columns and df["target"].nunique() < 20:
        # stratified sample
        return df.groupby("target").apply(
            lambda x: x.sample(frac=max_rows / len(df), random_state=42)
        ).reset_index(drop=True)

    # normal sample
    return df.sample(max_rows, random_state=42).reset_index(drop=True)


# ------------------------------------------------------------
# Stats helpers
# ------------------------------------------------------------

def describe_numeric(series: pd.Series) -> Dict:
    """
    Stats summary for numeric columns.
    """
    if series.empty:
        return {}

    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "25%": float(series.quantile(0.25)),
        "50%": float(series.quantile(0.50)),
        "75%": float(series.quantile(0.75)),
        "max": float(series.max())
    }


def describe_categorical(series: pd.Series) -> Dict:
    """
    Frequency summary for categoricals.
    """
    if series.empty:
        return {}

    freq = series.value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in freq.items()}


# ------------------------------------------------------------
# Datetime utilities
# ------------------------------------------------------------

def extract_datetime_features(series: pd.Series) -> pd.DataFrame:
    """
    Extracts rich temporal features from datetime columns.
    """
    if not np.issubdtype(series.dtype, np.datetime64):
        return pd.DataFrame()

    df = pd.DataFrame()
    df["year"] = series.dt.year
    df["month"] = series.dt.month
    df["day"] = series.dt.day
    df["weekday"] = series.dt.weekday
    df["hour"] = series.dt.hour
    df["is_weekend"] = series.dt.weekday >= 5
    return df


# ------------------------------------------------------------
# Correlation helpers
# ------------------------------------------------------------

def safe_corr(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Computes numeric correlation safely.
    Returns empty df if fail.
    """
    try:
        return df.corr(method=method)
    except Exception:
        return pd.DataFrame()


def cramers_v(confusion_matrix: pd.DataFrame) -> float:
    """
    Cramer's V for categorical-categorical correlations.
    """
    import scipy.stats as stats

    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


# ------------------------------------------------------------
# Outlier helpers
# ------------------------------------------------------------

def iqr_bounds(series: pd.Series) -> Tuple[float, float]:
    """
    Returns lower and upper bounds using IQR.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

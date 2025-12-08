# core/eda/missing.py
"""
Enterprise-grade missingness analysis module for AutoDataset-Lab.

Provides:
- column_missing_stats(df)
- row_missing_stats(df)
- missingness_correlation(df)
- missing_patterns(df)
- detect_missing_mechanisms(df)
- summarize_missingness(df)

Outputs are defensive, JSON-safe, and designed to feed:
- 2_Explore_Data.py numerics
- visualize_missing_heatmap()
- recommendations module
- PDF report generator
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any


# -----------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


# -----------------------------------------------------------
# Column-wise missing stats
# -----------------------------------------------------------
def column_missing_stats(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    total = len(df)
    out = {
        "column": [],
        "missing_count": [],
        "missing_percent": [],
        "dtype": [],
        "unique_non_null": []
    }

    for col in df.columns:
        ser = df[col]
        mc = int(ser.isna().sum())
        pct = round((mc / total) * 100, 4) if total else None

        out["column"].append(col)
        out["missing_count"].append(mc)
        out["missing_percent"].append(pct)
        out["dtype"].append(str(ser.dtype))
        try:
            out["unique_non_null"].append(int(ser.nunique(dropna=True)))
        except Exception:
            out["unique_non_null"].append(None)

    return pd.DataFrame(out)


# -----------------------------------------------------------
# Row-wise missing stats
# -----------------------------------------------------------
def row_missing_stats(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    total_cols = df.shape[1]
    missing_counts = df.isna().sum(axis=1)
    missing_pct = (missing_counts / total_cols) * 100

    return pd.DataFrame({
        "row_index": df.index,
        "missing_count": missing_counts,
        "missing_percent": missing_pct.round(4)
    })


# -----------------------------------------------------------
# Missingness correlation (co-missing patterns)
# -----------------------------------------------------------
def missingness_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes correlation matrix for missingness masks.
    This clusters columns that tend to be missing together.
    """
    if df.shape[1] > 2000:
        # Too many columns → avoid generating 2000x2000 matrix
        return pd.DataFrame()

    miss_mask = df.isna().astype(int)
    corr = miss_mask.corr().fillna(0)
    corr.index = corr.columns
    return corr


# -----------------------------------------------------------
# Frequent missingness patterns
# -----------------------------------------------------------
def missing_patterns(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Groups rows by their exact missingness pattern.
    """
    if df.empty:
        return pd.DataFrame()

    mask = df.isna().astype(int)
    pattern_strings = mask.apply(lambda row: "".join(row.astype(str)), axis=1)

    vc = pattern_strings.value_counts().head(top_n)

    return pd.DataFrame({
        "pattern": vc.index,
        "rows": vc.values
    })


# -----------------------------------------------------------
# Heuristic detection of missingness type (MCAR/MAR/MNAR)
# -----------------------------------------------------------
def detect_missing_mechanisms(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Very lightweight heuristics:
    MCAR — missingness shows no relation with other features.
    MAR — missingness correlated with other columns.
    MNAR — suspicious patterns like missing only for extreme values.
    """

    out = {
        "MCAR_like": [],
        "MAR_like": [],
        "MNAR_like": []
    }

    miss_mask = df.isna().astype(int)

    # For each column check correlation of is-missing with other columns
    for col in df.columns:
        ser = miss_mask[col]
        if ser.sum() == 0 or ser.sum() == len(df):
            continue  # nothing missing OR fully missing → skip

        correlations = {}
        for other in df.columns:
            if other == col:
                continue
            try:
                correlations[other] = abs(ser.corr(miss_mask[other]))
            except Exception:
                correlations[other] = 0

        max_corr = max(correlations.values()) if correlations else 0

        if max_corr < 0.05:
            out["MCAR_like"].append(col)
        elif max_corr < 0.30:
            out["MAR_like"].append(col)
        else:
            out["MNAR_like"].append(col)

    return out


# -----------------------------------------------------------
# Final combined missingness summary (enterprise-safe)
# -----------------------------------------------------------
def summarize_missingness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns the aggregated missingness package for numerics + UI + PDF.
    Also provides backwards-compatible `per_column` for the Streamlit UI.
    """

    # Compute components
    try:
        col_stats = column_missing_stats(df)
    except Exception:
        col_stats = pd.DataFrame()

    try:
        row_stats = row_missing_stats(df)
    except Exception:
        row_stats = pd.DataFrame()

    try:
        corr = missingness_correlation(df)
    except Exception:
        corr = pd.DataFrame()

    try:
        patterns = missing_patterns(df)
    except Exception:
        patterns = pd.DataFrame()

    try:
        mech = detect_missing_mechanisms(df)
    except Exception:
        mech = {"MCAR_like": [], "MAR_like": [], "MNAR_like": []}

    # Build old expected structure for Streamlit UI
    per_column = {}
    try:
        for _, row in col_stats.iterrows():
            per_column[row["column"]] = {
                "missing": int(row["missing_count"]),
                "missing_percent": float(row["missing_percent"]),
                "dtype": row["dtype"],
                "unique_non_null": int(row["unique_non_null"])
            }
    except Exception:
        per_column = {}

    # Text summary
    try:
        top_missing = col_stats.nlargest(5, "missing_percent")[["column", "missing_percent"]].to_dict(orient="records")
    except Exception:
        top_missing = []

    text_summary = {
        "top_missing_columns": top_missing,
        "missing_mechanisms": mech,
        "explanatory_text": _generate_explanatory_text(top_missing, mech)
    }

    # Final output with backwards compatibility
    return {
        "per_column": per_column,                   # <--- OLD FORMAT SO UI WORKS
        "column_stats": col_stats.to_dict(orient="records"),
        "row_stats_head": row_stats.head(20).to_dict(orient="records"),
        "missingness_correlation": corr.to_dict(),
        "patterns": patterns.to_dict(orient="records"),
        "mechanisms": mech,
        "text_summary": text_summary
    }



# -----------------------------------------------------------
# Human-readable narrative (LLM-quality)
# -----------------------------------------------------------
def _generate_explanatory_text(top_missing, mechanisms):
    lines = []

    if not top_missing:
        return "No significant missingness detected across the dataset."

    lines.append("Detected notable missingness patterns in the dataset:")

    for item in top_missing:
        lines.append(f"- {item['column']} shows {item['missing_percent']}% missing values.")

    if mechanisms.get("MCAR_like"):
        lines.append("\nColumns likely Missing Completely At Random (MCAR): " +
                     ", ".join(mechanisms["MCAR_like"]))

    if mechanisms.get("MAR_like"):
        lines.append("\nColumns showing potential MAR behavior (correlated missingness): " +
                     ", ".join(mechanisms["MAR_like"]))

    if mechanisms.get("MNAR_like"):
        lines.append("\nColumns with suspicious MNAR-like patterns (likely not random): " +
                     ", ".join(mechanisms["MNAR_like"]))

    return "\n".join(lines)

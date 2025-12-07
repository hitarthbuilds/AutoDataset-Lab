# core/eda/quality.py
"""
Enterprise-grade data quality diagnostics for AutoDataset-Lab.

Functions:
- detect_duplicates(df)
- detect_constant_columns(df, unique_threshold=1)
- detect_infinities_and_negatives(df)
- compute_cardinality(df, high_card_threshold=0.5)
- compute_data_quality_report(df, sample_size=10, high_card_threshold=0.5)

Design goals:
- Defensive: checks types and handles edge cases gracefully.
- JSON-safe outputs: only basic python types in the returned dict.
- Useful for UI, PDF report, recommendations, and LLM prompt generation.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math
import numpy as np
import pandas as pd

# ---------- Helpers ----------
def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _is_infinite_series(s: pd.Series) -> pd.Series:
    """Return boolean mask for infinite values in a Series (works for numeric types)."""
    try:
        return np.isinf(s.astype(float))
    except Exception:
        # For non-convertible types, nothing is infinite
        return pd.Series([False] * len(s), index=s.index)


def _sample_records(df: pd.DataFrame, n: int = 10) -> List[Dict[str, Any]]:
    try:
        if df is None or df.empty:
            return []
        n = min(max(1, n), len(df))
        sample_df = df.head(n) if len(df) <= n else df.sample(n=n, random_state=1)
        return sample_df.fillna("").astype(str).to_dict(orient="records")
    except Exception:
        return []


# ---------- Low-level detectors ----------
def detect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect duplicate rows. Returns count and sample indexes (limited)."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    try:
        dup_mask = df.duplicated(keep="first")
        dup_count = int(dup_mask.sum())
        dup_idx = df.index[dup_mask].tolist()[:50]  # only return first 50 indices
        return {"duplicate_count": dup_count, "duplicate_index_sample": dup_idx}
    except Exception:
        return {"duplicate_count": None, "duplicate_index_sample": []}


def detect_constant_columns(df: pd.DataFrame, unique_threshold: int = 1) -> Dict[str, Any]:
    """
    Columns with <= unique_threshold non-null distinct values are considered constant-ish.
    Returns list of constant columns and counts.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    const_cols = []
    try:
        for col in df.columns:
            try:
                nunique = int(df[col].nunique(dropna=True))
            except Exception:
                nunique = None

            if nunique is not None and nunique <= unique_threshold:
                const_cols.append({"column": col, "unique_non_null": nunique})
    except Exception:
        pass

    return {"constant_columns": const_cols, "constant_count": len(const_cols)}


def detect_infinities_and_negatives(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect infinities and negative numbers for numeric columns.
    Returns per-column counts.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    infinities = []
    negatives = []
    zeros = []
    try:
        for col in df.columns:
            ser = df[col]
            # attempt numeric conversion (coerce non-numeric to NaN)
            try:
                num = pd.to_numeric(ser, errors="coerce")
            except Exception:
                num = None

            inf_count = 0
            neg_count = 0
            zero_count = 0
            if num is not None:
                # Count infinities
                try:
                    inf_mask = np.isinf(num.values)
                    inf_count = int(np.sum(inf_mask))
                except Exception:
                    inf_count = 0
                # negatives
                try:
                    neg_mask = (~np.isnan(num.values)) & (num.values < 0)
                    neg_count = int(np.sum(neg_mask))
                except Exception:
                    neg_count = 0
                # zeros
                try:
                    zero_mask = (~np.isnan(num.values)) & (num.values == 0)
                    zero_count = int(np.sum(zero_mask))
                except Exception:
                    zero_count = 0

            infinities.append({"column": col, "infinite_count": inf_count})
            negatives.append({"column": col, "negative_count": neg_count})
            zeros.append({"column": col, "zero_count": zero_count})
    except Exception:
        pass

    return {"infinities": infinities, "negatives": negatives, "zeros": zeros}


def compute_cardinality(df: pd.DataFrame, high_card_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute cardinality and flag high-cardinality columns.
    high_card_threshold can be fraction (e.g., 0.5 = 50% of rows considered high cardinality).
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    total_rows = len(df)
    card_list = []
    high_card_cols = []
    try:
        for col in df.columns:
            try:
                nunique = int(df[col].nunique(dropna=True))
            except Exception:
                nunique = None
            pct = None
            try:
                pct = round((nunique / total_rows) * 100, 4) if total_rows and nunique is not None else None
            except Exception:
                pct = None

            item = {"column": col, "unique_non_null": nunique, "unique_percent": pct}
            card_list.append(item)
            if pct is not None and pct >= (high_card_threshold * 100):
                high_card_cols.append(item)
    except Exception:
        pass

    return {"cardinality": card_list, "high_cardinality_columns": high_card_cols}


# ---------- Quality scoring ----------
def _compute_quality_score(metrics: Dict[str, Any]) -> float:
    """
    Simple heuristic score (0..100). We combine:
    - duplicate rate
    - fraction of constant columns
    - fraction of columns with infinities or many negatives
    - high-cardinality issues are not penalized strongly (context-dependent)
    This is intentionally conservative. Fine-tune weights as needed.
    """
    # defaults
    try:
        total_rows = metrics.get("rows") or 0
        total_cols = metrics.get("columns") or 0
        dup_count = metrics.get("duplicate_rows", 0) or 0
        constant_count = metrics.get("constant_count", 0) or 0

        inf_list = metrics.get("infinities", [])
        neg_list = metrics.get("negatives", [])

        # fraction metrics
        dup_frac = (dup_count / total_rows) if total_rows else 0.0
        const_frac = (constant_count / total_cols) if total_cols else 0.0

        # columns with issues
        inf_cols = sum(1 for x in inf_list if x.get("infinite_count", 0) > 0)
        neg_cols = sum(1 for x in neg_list if x.get("negative_count", 0) > 0)

        inf_frac = (inf_cols / total_cols) if total_cols else 0.0
        neg_frac = (neg_cols / total_cols) if total_cols else 0.0

        # weights (tunable)
        w_dup = 0.30
        w_const = 0.25
        w_inf = 0.25
        w_neg = 0.20

        raw = 1.0 - (w_dup * dup_frac + w_const * const_frac + w_inf * inf_frac + w_neg * neg_frac)
        score = max(0.0, min(1.0, raw))  # clamp
        return round(score * 100, 2)
    except Exception:
        return 50.0


# ---------- Top-level report builder ----------
def compute_data_quality_report(
    df: pd.DataFrame,
    sample_size: int = 10,
    unique_threshold: int = 1,
    high_card_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute a comprehensive data-quality dictionary for UI / PDF / recommendations.

    Returns JSON-friendly dict with:
    - rows, columns
    - duplicates summary
    - constant columns
    - infinities / negatives / zeros
    - cardinality and high-cardinality
    - per-column quick stats (dtype, nulls, percent null, sample unique)
    - simple quality_score (0..100)
    - sample rows
    - suggested quick fixes (text)
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    rows = len(df)
    cols = df.shape[1]

    # top-level
    try:
        dup_info = detect_duplicates(df)
    except Exception:
        dup_info = {"duplicate_count": None, "duplicate_index_sample": []}

    try:
        const_info = detect_constant_columns(df, unique_threshold=unique_threshold)
    except Exception:
        const_info = {"constant_columns": [], "constant_count": 0}

    try:
        inf_neg_info = detect_infinities_and_negatives(df)
    except Exception:
        inf_neg_info = {"infinities": [], "negatives": [], "zeros": []}

    try:
        card_info = compute_cardinality(df, high_card_threshold=high_card_threshold)
    except Exception:
        card_info = {"cardinality": [], "high_cardinality_columns": []}

    # per-column quick stats
    per_column = []
    try:
        for col in df.columns:
            try:
                ser = df[col]
                total = rows
                nulls = int(ser.isna().sum())
                null_pct = round((nulls / total) * 100, 4) if total else None
                dtype = str(ser.dtype)
                try:
                    unique_non_null = int(ser.nunique(dropna=True))
                except Exception:
                    unique_non_null = None
                try:
                    n_infinite = int(np.sum(np.isinf(pd.to_numeric(ser, errors="coerce").values)))
                except Exception:
                    n_infinite = 0
                try:
                    n_negative = int(np.sum((pd.to_numeric(ser, errors="coerce") < 0).fillna(False)))
                except Exception:
                    n_negative = 0

                per_column.append({
                    "column": col,
                    "dtype": dtype,
                    "null_count": nulls,
                    "null_percent": null_pct,
                    "unique_non_null": unique_non_null,
                    "infinite_count": n_infinite,
                    "negative_count": n_negative
                })
            except Exception:
                per_column.append({"column": col})
    except Exception:
        per_column = []

    # sample rows
    sample = _sample_records(df, n=sample_size)

    # suggested quick fixes (high-level human text)
    suggested_fixes = []
    # duplicates
    if dup_info.get("duplicate_count"):
        suggested_fixes.append("Remove exact duplicate rows or deduplicate on a set of business keys.")
    # constants
    if const_info.get("constant_count"):
        suggested_fixes.append("Drop constant columns (no variance) or use them only for metadata mapping.")
    # infinities
    inf_cols = [x["column"] for x in inf_neg_info.get("infinities", []) if x.get("infinite_count", 0) > 0]
    if inf_cols:
        suggested_fixes.append(f"Columns with infinities detected: {', '.join(inf_cols)}. Replace with NaN or clip values.")
    # negatives
    neg_cols = [x["column"] for x in inf_neg_info.get("negatives", []) if x.get("negative_count", 0) > 0]
    if neg_cols:
        suggested_fixes.append(f"Columns with negative values (check business logic): {', '.join(neg_cols)}.")
    # high cardinality
    high_card_cols = [c["column"] for c in card_info.get("high_cardinality_columns", [])][:10]
    if high_card_cols:
        suggested_fixes.append(f"High-cardinality categorical columns: {', '.join(high_card_cols)}. Consider hashing/frequency or target encoding.")

    # quality score
    metrics_for_score = {
        "rows": rows,
        "columns": cols,
        "duplicate_rows": dup_info.get("duplicate_count", 0) or 0,
        "constant_count": const_info.get("constant_count", 0) or 0,
        "infinities": inf_neg_info.get("infinities", []) or [],
        "negatives": inf_neg_info.get("negatives", []) or []
    }
    quality_score = _compute_quality_score(metrics_for_score)

    # final report
    report = {
        "rows": int(rows),
        "columns": int(cols),
        "duplicate_summary": dup_info,
        "constant_summary": const_info,
        "infinities_negatives": inf_neg_info,
        "cardinality": card_info,
        "per_column": per_column,
        "quality_score": quality_score,
        "suggested_fixes": suggested_fixes,
        "sample": sample
    }

    return report


# When run as script — tiny smoke check (won't run in your app)
if __name__ == "__main__":
    # small demo
    df_demo = pd.DataFrame({
        "a": [1, 2, 3, np.nan, np.inf, -1],
        "b": ["x", "x", "x", "x", "x", "x"],
        "c": [None, None, None, None, None, None]
    })
    r = compute_data_quality_report(df_demo)
    import json
    print(json.dumps(r, indent=2))

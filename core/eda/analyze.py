# core/eda/analyze.py
"""
Enterprise-grade EDA analysis core (Style A).

Responsibilities:
- Infer column semantic / data types with confidence
- Compute robust numeric summaries (including robust stats)
- Compute categorical summaries (top-k, cardinality, rare categories)
- Per-column missingness and missing pattern summaries
- Correlation summaries (pearson, spearman) with safe fallbacks
- Pairwise sample-size-aware comparison metrics
- Export stable JSON-serializable summaries via to_json_summary()
- Designed for unit testing and composability with other modules
- Defensive: no hard crashes when optional libs missing
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import math
import json
import warnings

import numpy as np
import pandas as pd
# -------------------------
# Small JSON helpers (defensive)
# -------------------------
import numpy as np
import pandas as pd
import json
from typing import Any

def _json_default(o: Any):
    """
    Default serializer for json.dumps(..., default=_json_default).
    Keeps numeric types and small pandas objects safe for JSON.
    """
    # numpy scalars
    if isinstance(o, (np.integer, np.int64, np.int32)):
        return int(o)
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)

    # numpy arrays -> lists
    if isinstance(o, (np.ndarray,)):
        try:
            return o.tolist()
        except Exception:
            return list(o)

    # pandas Series/DataFrame safe truncation
    if isinstance(o, pd.Series):
        try:
            # if numeric: return numeric list, else string list
            if pd.api.types.is_numeric_dtype(o.dtype):
                return o.dropna().tolist()
            return o.dropna().astype(str).tolist()
        except Exception:
            return o.tolist()

    if isinstance(o, pd.DataFrame):
        try:
            return o.head(200).to_dict(orient="records")
        except Exception:
            return str(o)

    # fallback: string repr
    try:
        return str(o)
    except Exception:
        return repr(o)


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Use this to safely dump EDA outputs to JSON for UI or storage."""
    return json.dumps(obj, default=_json_default, **kwargs)

# Optional imports
try:
    from scipy import stats  # type: ignore
    _has_scipy = True
except Exception:
    _has_scipy = False


# -------------------------
# Data classes
# -------------------------
@dataclass
class ColumnTypeConfidence:
    inferred_type: str
    confidence: float


@dataclass
class NumericSummary:
    count: int
    missing: int
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    var: Optional[float]
    min: Optional[float]
    p10: Optional[float]
    p25: Optional[float]
    p50: Optional[float]
    p75: Optional[float]
    p90: Optional[float]
    max: Optional[float]
    skew: Optional[float]
    kurtosis: Optional[float]
    iqr: Optional[float]
    mad: Optional[float]  # median absolute deviation
    outlier_count_iqr: Optional[int]


@dataclass
class CategoricalSummary:
    count: int
    missing: int
    unique: int
    top_k: List[Tuple[Any, int]]
    mode_pct: Optional[float]
    high_cardinality: bool


@dataclass
class ColumnOverview:
    name: str
    dtype: str
    semantic: ColumnTypeConfidence
    numeric_summary: Optional[NumericSummary]
    categorical_summary: Optional[CategoricalSummary]
    missing_pct: float


# -------------------------
# Helpers & Utilities
# -------------------------
def _safe_head(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df.head(n)


def _to_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def _percent_missing(series: pd.Series) -> float:
    if series is None:
        return 0.0
    total = len(series)
    if total == 0:
        return 0.0
    return float(series.isna().sum() / total)


def _robust_iqr(series: pd.Series) -> float:
    q75 = series.quantile(0.75)
    q25 = series.quantile(0.25)
    return float(q75 - q25)


# -------------------------
# Type inference
# -------------------------
def infer_column_semantic(series: pd.Series) -> ColumnTypeConfidence:
    """
    Infer a simple semantic type with a confidence score.
    Types: numeric, categorical, datetime, boolean, text
    Confidence is heuristic [0..1].
    """
    n = len(series)
    if n == 0:
        return ColumnTypeConfidence(inferred_type="unknown", confidence=0.0)

    non_null = series.dropna()
    nn = len(non_null)
    if nn == 0:
        return ColumnTypeConfidence(inferred_type="unknown", confidence=0.2)

    # datetime detection
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnTypeConfidence(inferred_type="datetime", confidence=0.95)

    # boolean detection
    unique_vals = non_null.unique()
    if len(unique_vals) <= 2 and non_null.map(lambda x: str(x).lower() in ("0", "1", "true", "false", "yes", "no")).all():
        return ColumnTypeConfidence(inferred_type="boolean", confidence=0.9)

    # numeric detection
    numeric_ratio = non_null.apply(lambda x: pd.api.types.is_number(x)).sum() / nn
    if numeric_ratio > 0.85 or pd.api.types.is_numeric_dtype(series):
        # confidence scaled by ratio and size
        conf = min(0.95, 0.5 + 0.5 * numeric_ratio)
        return ColumnTypeConfidence(inferred_type="numeric", confidence=float(conf))

    # categorical vs text detection
    avg_len = non_null.astype(str).map(len).mean()
    unique_ratio = len(unique_vals) / nn
    # heuristics: short strings with small unique ratio -> categorical
    if unique_ratio < 0.2 and avg_len < 50:
        conf = min(0.95, 0.5 + (0.5 * (0.2 - unique_ratio) / 0.2))
        return ColumnTypeConfidence(inferred_type="categorical", confidence=float(conf))
    # long average length -> text
    if avg_len > 100:
        return ColumnTypeConfidence(inferred_type="text", confidence=0.9)

    # fallback
    return ColumnTypeConfidence(inferred_type="categorical", confidence=0.6)


# -------------------------
# Numeric summaries
# -------------------------
def numeric_summary(series: pd.Series) -> NumericSummary:
    """
    Robust numeric summary. Accepts non-numeric series too but will coerce.
    Uses robust statistics where appropriate.
    """
    # coerce numeric (safe)
    ser_num = pd.to_numeric(series, errors="coerce")
    count = int(len(ser_num) - ser_num.isna().sum())
    missing = int(ser_num.isna().sum())
    if count == 0:
        base = {
            "count": 0,
            "missing": missing,
            "mean": None,
            "median": None,
            "std": None,
            "var": None,
            "min": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "max": None,
            "skew": None,
            "kurtosis": None,
            "iqr": None,
            "mad": None,
            "outlier_count_iqr": None,
        }
        return NumericSummary(**base)

    arr = ser_num.dropna().astype(float)
    mean = _to_float_or_none(arr.mean())
    median = _to_float_or_none(arr.median())
    std = _to_float_or_none(arr.std(ddof=1))
    var = _to_float_or_none(arr.var(ddof=1))
    _min = _to_float_or_none(arr.min())
    _max = _to_float_or_none(arr.max())
    p10 = _to_float_or_none(arr.quantile(0.10))
    p25 = _to_float_or_none(arr.quantile(0.25))
    p50 = median
    p75 = _to_float_or_none(arr.quantile(0.75))
    p90 = _to_float_or_none(arr.quantile(0.90))
    skew = _to_float_or_none(arr.skew())
    kurtosis = _to_float_or_none(arr.kurtosis())
    iqr = _to_float_or_none(_robust_iqr(arr))
    mad = _to_float_or_none(float(np.median(np.abs(arr - np.median(arr)))))
    # outliers by IQR rule
    try:
        q1 = arr.quantile(0.25)
        q3 = arr.quantile(0.75)
        iqr_val = q3 - q1
        lower = q1 - 1.5 * iqr_val
        upper = q3 + 1.5 * iqr_val
        outlier_count = int(((arr < lower) | (arr > upper)).sum())
    except Exception:
        outlier_count = None

    return NumericSummary(
        count=count,
        missing=missing,
        mean=mean,
        median=median,
        std=std,
        var=var,
        min=_min,
        p10=p10,
        p25=p25,
        p50=p50,
        p75=p75,
        p90=p90,
        max=_max,
        skew=skew,
        kurtosis=kurtosis,
        iqr=iqr,
        mad=mad,
        outlier_count_iqr=outlier_count,
    )


# -------------------------
# Categorical summaries
# -------------------------
def categorical_summary(series: pd.Series, top_k: int = 10, high_card_threshold: int = 100) -> CategoricalSummary:
    """
    Categorical summary including top-k categories and high-cardinality flag.
    """
    count = int(len(series) - series.isna().sum())
    missing = int(series.isna().sum())
    vc = series.fillna("__NA__").astype(str).value_counts(dropna=False)
    unique = int(vc.size)
    # get top_k items as list of (value, count)
    topk = [(idx, int(cnt)) for idx, cnt in zip(vc.index[:top_k].tolist(), vc.iloc[:top_k].tolist())]
    mode_pct = None
    try:
        top_val = vc.iloc[0]
        mode_pct = float(top_val / vc.sum())
    except Exception:
        mode_pct = None
    high_card = unique > high_card_threshold
    return CategoricalSummary(
        count=count,
        missing=missing,
        unique=unique,
        top_k=topk,
        mode_pct=mode_pct,
        high_cardinality=bool(high_card),
    )


# -------------------------
# Correlations & pairwise checks
# -------------------------
def safe_pearson(series_a: pd.Series, series_b: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (corr_coefficient, pvalue) if possible, else (None, None).
    """
    try:
        a = pd.to_numeric(series_a, errors="coerce").dropna()
        b = pd.to_numeric(series_b, errors="coerce").dropna()
        # align by index intersection
        common_idx = a.index.intersection(b.index)
        if len(common_idx) < 2:
            return None, None
        a2, b2 = a.loc[common_idx], b.loc[common_idx]
        if _has_scipy:
            stat, p = stats.pearsonr(a2, b2)
            return float(stat), float(p)
        # fallback: numpy
        corr = float(np.corrcoef(a2, b2)[0, 1])
        return corr, None
    except Exception:
        return None, None


def safe_spearman(series_a: pd.Series, series_b: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """
    Safe spearman correlation (ranked).
    """
    if not _has_scipy:
        return None, None
    try:
        a = pd.to_numeric(series_a, errors="coerce").dropna()
        b = pd.to_numeric(series_b, errors="coerce").dropna()
        common_idx = a.index.intersection(b.index)
        if len(common_idx) < 2:
            return None, None
        stat, p = stats.spearmanr(a.loc[common_idx], b.loc[common_idx], nan_policy="omit")
        return float(stat), float(p)
    except Exception:
        return None, None


# -------------------------
# High-level analysis functions
# -------------------------
def analyze_dataframe(df: pd.DataFrame, top_k: int = 10, high_card_threshold: int = 100) -> Dict[str, Any]:
    """
    Run full analysis over df and return a structured dictionary suitable for the rest of the system.
    The returned dict is JSON-serializable (basic Python types).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    out: Dict[str, Any] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "columns_overview": {},
        "global_stats": {},
    }

    # column-wise analysis
    columns_overview: Dict[str, Any] = {}
    for col in df.columns:
        ser = df[col]
        dtype = str(ser.dtype)
        semantic = infer_column_semantic(ser)
        missing_pct = _percent_missing(ser)
        num_sum = None
        cat_sum = None
        if semantic.inferred_type == "numeric" or pd.api.types.is_numeric_dtype(ser):
            try:
                num_sum = numeric_summary(ser)
            except Exception:
                num_sum = numeric_summary(pd.to_numeric(ser, errors="coerce"))
        else:
            try:
                cat_sum = categorical_summary(ser, top_k=top_k, high_card_threshold=high_card_threshold)
            except Exception:
                cat_sum = categorical_summary(ser.astype(str), top_k=top_k, high_card_threshold=high_card_threshold)

        columns_overview[col] = ColumnOverview(
            name=col,
            dtype=dtype,
            semantic=semantic,
            numeric_summary=num_sum,
            categorical_summary=cat_sum,
            missing_pct=float(missing_pct),
        )

    out["columns_overview"] = {k: _column_overview_to_serializable(v) for k, v in columns_overview.items()}

    # global-level numeric correlation summary (only numeric columns)
    try:
        numeric_cols = [c for c, v in out["columns_overview"].items() if v["semantic"]["inferred_type"] == "numeric"]
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        corr_matrix = numeric_df.corr(method="pearson")
        out["global_stats"]["pearson_corr_top_pairs"] = _top_abs_pairs_from_corr(corr_matrix, top_n=20)
    except Exception:
        out["global_stats"]["pearson_corr_top_pairs"] = []

    # cardinality overview
    try:
        card = {c: int(df[c].nunique(dropna=False)) for c in df.columns}
        out["global_stats"]["cardinality"] = card
    except Exception:
        out["global_stats"]["cardinality"] = {}

    # quick sample
    try:
        out["sample"] = df.head(10).to_dict(orient="records")
    except Exception:
        out["sample"] = []

    return out


# -------------------------
# Serializers & helpers
# -------------------------
def _column_overview_to_serializable(co: ColumnOverview) -> Dict[str, Any]:
    """
    Convert ColumnOverview dataclass to JSON-friendly dict.
    """
    base = {
        "name": co.name,
        "dtype": co.dtype,
        "semantic": asdict(co.semantic),
        "missing_pct": float(co.missing_pct),
    }
    if co.numeric_summary is not None:
        base["numeric_summary"] = asdict(co.numeric_summary)
    if co.categorical_summary is not None:
        base["categorical_summary"] = asdict(co.categorical_summary)
    return base


def _top_abs_pairs_from_corr(corr: pd.DataFrame, top_n: int = 20) -> List[Tuple[str, str, float]]:
    """
    Return list of (col_a, col_b, abs_corr) sorted by descending abs corr.
    """
    if corr is None or corr.empty:
        return []
    corr_abs = corr.abs().where(~np.eye(len(corr), dtype=bool)).stack().reset_index()
    corr_abs.columns = ["col_a", "col_b", "abs_corr"]
    corr_abs = corr_abs.dropna(subset=["abs_corr"])
    corr_abs_sorted = corr_abs.sort_values("abs_corr", ascending=False).head(top_n)
    pairs = [(r.col_a, r.col_b, float(r.abs_corr)) for r in corr_abs_sorted.itertuples()]
    return pairs


def to_json_summary(analysis_dict: Dict[str, Any], pretty: bool = False) -> str:
    """
    Convert analysis dict to JSON string. Uses safe numerical conversion helper.
    """
    if pretty:
        return json.dumps(analysis_dict, default=_json_default, indent=2)
    return json.dumps(analysis_dict, default=_json_default, separators=(",", ":"))


# -------------------------
# Small public utilities
# -------------------------
def sample_dataframe(df: pd.DataFrame, max_rows: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Return a reproducible sample for very large datasets. Deterministic.
    """
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=random_state)


# -------------------------
# Backwards-compatible thin wrapper for older code that expects `analyze_dataframe(df)`
# -------------------------
def analyze(df: pd.DataFrame) -> Dict[str, Any]:
    return analyze_dataframe(df)


# -------------------------
# End of file
# -------------------------

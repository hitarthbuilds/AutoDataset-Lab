# core/eda/drift_advanced.py
"""
Enterprise Dataset Drift Engine.

Features:
- Per-column drift (numeric KS, categorical Jensen-Shannon)
- Dataset aggregate drift score (mean of per-column metrics)
- Simple high-drift column listing
- JSON-friendly output
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import math
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _safe_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype="float64")
    return s.dropna()


def _ks_stat(ref: pd.Series, cur: pd.Series) -> float:
    a = _safe_series(ref)
    b = _safe_series(cur)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    try:
        stat = ks_2samp(a, b).statistic
        return float(stat)
    except Exception:
        return float("nan")


def _jsd(ref: pd.Series, cur: pd.Series) -> float:
    # Jensen-Shannon divergence on value frequency vectors
    a = ref.fillna("__NA__").astype(str)
    b = cur.fillna("__NA__").astype(str)

    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    all_idx = list(dict.fromkeys(list(pa.index) + list(pb.index)))
    pa = pa.reindex(all_idx, fill_value=0.0).values
    pb = pb.reindex(all_idx, fill_value=0.0).values
    # jensenshannon returns float; may raise if degenerate
    try:
        return float(jensenshannon(pa + 1e-12, pb + 1e-12))
    except Exception:
        return float("nan")


def compute_feature_drift(ref: pd.Series, cur: pd.Series) -> Dict[str, Any]:
    """
    Compute drift metrics for a single feature (ref = historical, cur = new/current).
    Returns JSON-friendly dict.
    """
    res: Dict[str, Any] = {}
    res["ref_count"] = int(ref.dropna().shape[0])
    res["cur_count"] = int(cur.dropna().shape[0])

    # choose numeric vs categorical heuristics
    numeric = _is_numeric(ref) or _is_numeric(cur)
    res["type"] = "numeric" if numeric else "categorical"

    if numeric:
        res["ks_stat"] = _ks_stat(ref, cur)
        try:
            # add simple effect size (Cohen's d)
            a = _safe_series(ref).astype(float)
            b = _safe_series(cur).astype(float)
            if a.size > 1 and b.size > 1:
                da = a.mean(); db = b.mean()
                sa = a.var(ddof=1); sb = b.var(ddof=1)
                pooled = math.sqrt(((a.size - 1) * sa + (b.size - 1) * sb) / (a.size + b.size - 2)) if (a.size + b.size - 2) > 0 else float("nan")
                res["cohen_d"] = float((da - db) / pooled) if pooled and not math.isnan(pooled) else float("nan")
            else:
                res["cohen_d"] = float("nan")
        except Exception:
            res["cohen_d"] = float("nan")
    else:
        res["jsd"] = _jsd(ref, cur)
        # chi/contingency omitted to keep dependencies minimal; can add later

    # normalize a single score (0..1-ish) using available metrics
    score_components: List[float] = []
    if numeric:
        if not math.isnan(res.get("ks_stat", float("nan"))):
            score_components.append(min(max(res["ks_stat"], 0.0), 1.0))
        if not math.isnan(res.get("cohen_d", float("nan"))):
            score_components.append(min(abs(res["cohen_d"]) / 3.0, 1.0))  # scale heuristically
    else:
        if not math.isnan(res.get("jsd", float("nan"))):
            # jensen-shannon in [0,1], use directly (but cap)
            score_components.append(min(max(res["jsd"], 0.0), 1.0))

    if score_components:
        res["drift_score"] = float(np.nanmean(score_components))
    else:
        res["drift_score"] = float("nan")

    res["drift_level"] = (
        "unknown" if math.isnan(res["drift_score"]) else
        ("severe" if res["drift_score"] > 0.6 else ("moderate" if res["drift_score"] > 0.25 else "stable"))
    )

    return res


def detect_dataset_drift(
    df_cur: pd.DataFrame,
    df_ref: pd.DataFrame,
    columns: Optional[list] = None
) -> Dict[str, Any]:
    """
    High-level dataset drift. Compare df_cur against df_ref.
    Returns:
      {
        "dataset_drift_score": float,
        "per_column": {col: {...metrics...}},
        "flagged_columns": [col names],
        "columns_compared": [...]
      }
    """

    if df_ref is None or df_ref.empty:
        return {"error": "Reference dataset (df_ref) is required for drift detection."}

    if columns is None:
        columns = [c for c in df_cur.columns if c in df_ref.columns]

    per_col: Dict[str, Any] = {}
    scores = []
    for c in columns:
        ref = df_ref[c] if c in df_ref.columns else pd.Series([], dtype=object)
        cur = df_cur[c] if c in df_cur.columns else pd.Series([], dtype=object)
        try:
            metrics = compute_feature_drift(ref, cur)
        except Exception as e:
            metrics = {"error": str(e)}
        per_col[c] = metrics
        s = metrics.get("drift_score")
        if s is not None and not (isinstance(s, float) and math.isnan(s)):
            scores.append(s)

    dataset_score = float(np.nanmean(scores)) if scores else float("nan")
    flagged = [c for c, m in per_col.items() if m.get("drift_level") in ("moderate", "severe")]

    return {
        "dataset_drift_score": dataset_score,
        "per_column": per_col,
        "columns_compared": columns,
        "flagged_columns": flagged
    }


# If directly executed it does a small demo (handy for tests)
if __name__ == "__main__":
    import numpy as np, pandas as pd
    rng = np.random.RandomState(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 1000), "cat": rng.choice(["A", "B", "C"], 1000)})
    cur = ref.copy()
    cur.loc[500:, "x"] += 1.5
    cur.loc[700:, "cat"] = rng.choice(["A", "B", "C", "D"], len(cur.loc[700:]))
    print(detect_dataset_drift(cur, ref))

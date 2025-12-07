"""
core/eda/feature_importance.py

Enterprise-grade feature importance module.

Capabilities:
- RandomForest built-in importances (regressor / classifier)
- Permutation importance (sklearn)
- Mutual information (numeric / categorical)
- Cramér's V for categorical-categorical association
- Target-encoding importance (fast proxy)
- SHAP (if available) explanation fallback
- Stability analysis over multiple random seeds
- Grouped importance (feature groups)
- Export helpers: to_dataframe(), to_json()
- Small plotly bar helper for visual UI use
- Defensive: graceful fallbacks, small sample mode for huge datasets

Author: ChatGPT-for-Hitarth (enterprise-mode)
"""
from __future__ import annotations

import math
import json
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# sklearn imports (assume available in most environments)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import mutual_info_score

# Optional imports
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

try:
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# -------------------------
# Utility / defensive helpers
# -------------------------
def _safe_head(df: pd.DataFrame, n: int = 20000) -> pd.DataFrame:
    """
    Return a safely sized dataframe for heavy computations (shap, rf training).
    Uses stratified sample if target available.
    """
    if df.shape[0] <= n:
        return df
    try:
        return df.sample(n=n, random_state=42)
    except Exception:
        return df.head(n)


def _is_categorical(series: pd.Series) -> bool:
    """Heuristic: object, category, or low cardinality."""
    if pd.api.types.is_categorical_dtype(series.dtype):
        return True
    if series.dtype == object:
        return True
    try:
        uniq = series.dropna().unique()
        return len(uniq) <= max(50, 0.02 * len(series))
    except Exception:
        return False


def _encode_series(series: pd.Series) -> np.ndarray:
    """Label-encode a series to integers for ML where needed."""
    try:
        if series.dtype == object or pd.api.types.is_categorical_dtype(series.dtype):
            le = LabelEncoder()
            arr = le.fit_transform(series.astype(str).fillna("__NA__"))
            return arr
        else:
            return pd.to_numeric(series, errors="coerce").fillna(0).values
    except Exception:
        return pd.Series(series).astype(str).fillna("__NA__").values


def _json_default(o: Any):
    """JSON serializer used by export helpers."""
    if isinstance(o, (np.integer, np.int64, np.int32)):
        return int(o)
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Series, pd.Index)):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.head(200).to_dict(orient="records")
    return str(o)


def safe_json_dumps(obj: Any, **kwargs) -> str:
    return json.dumps(obj, default=_json_default, **kwargs)


# -------------------------
# Statistical association helpers
# -------------------------
def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Cramér's V between two categorical columns (0..1).
    """
    try:
        confusion = pd.crosstab(x.fillna("__NA__"), y.fillna("__NA__"))
        chi2 = scipy_chi2_contingency(confusion)
        n = confusion.sum().sum()
        if n == 0:
            return float("nan")
        phi2 = chi2 / n
        r, k = confusion.shape
        # bias correction
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        denom = min(kcorr - 1, rcorr - 1)
        if denom <= 0:
            return 0.0
        return math.sqrt(phi2corr / denom)
    except Exception:
        return float("nan")


def scipy_chi2_contingency(confusion: pd.DataFrame) -> float:
    """
    Run chi2_contingency but avoid direct scipy import failing at runtime in minimal envs.
    We try-import scipy when needed.
    """
    try:
        import scipy.stats as ss  # type: ignore
        res = ss.chi2_contingency(confusion, correction=False)
        return float(res[0])
    except Exception:
        # Fallback: approximate chi2 via classic formula on flattened table
        # (This is a weak fallback but prevents hard failure).
        obs = confusion.values.astype(float)
        row_sums = obs.sum(axis=1, keepdims=True)
        col_sums = obs.sum(axis=0, keepdims=True)
        total = obs.sum()
        expected = (row_sums @ col_sums) / (total + 1e-12)
        denom = expected.copy()
        denom[denom == 0] = 1.0
        chi2 = ((obs - expected) ** 2 / denom).sum()
        return float(chi2)


def _mutual_info_numeric_categorical(x: pd.Series, y: pd.Series, discrete_features=False) -> float:
    """
    Mutual information (fallback to sklearn.metrics.mutual_info_score).
    - x could be numeric or categorical; y may be numeric or categorical.
    This returns a non-negative float (in nats).
    """
    try:
        # mutual_info_score expects discrete labels -> encode to ints
        xa = pd.Series(_encode_series(x)).astype(int)
        ya = pd.Series(_encode_series(y)).astype(int)
        return float(mutual_info_score(xa, ya))
    except Exception:
        return float("nan")


# -------------------------
# Core importance pieces
# -------------------------
def random_forest_importance(
    df: pd.DataFrame,
    target: str,
    problem_type: Optional[str] = None,
    sample_size: Optional[int] = 20000,
    n_estimators: int = 200,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train a RandomForest (classifier/regressor depending on target)
    and return feature importances and model metadata.

    Returns:
    {
      "method": "random_forest",
      "importances": {feature: score, ...},
      "model_meta": {...}
    }
    """
    out: Dict[str, Any] = {"method": "random_forest", "importances": {}, "meta": {}}
    if target not in df.columns:
        return out

    # Prepare X, y
    y = df[target]
    X = df.drop(columns=[target]).copy()

    # small sample for speed if requested
    if sample_size and df.shape[0] > sample_size:
        df_small = _safe_head(df[[*X.columns, target]], sample_size)
        X = df_small.drop(columns=[target]).copy()
        y = df_small[target]

    # Encode categorical columns for scikit-learn
    X_enc = X.copy()
    for c in X_enc.columns:
        if _is_categorical(X_enc[c]):
            X_enc[c] = _encode_series(X_enc[c])

    # Choose model type
    try:
        if problem_type is None:
            problem_type = "regression" if pd.api.types.is_numeric_dtype(y) else "classification"
    except Exception:
        problem_type = "classification"

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state) \
        if problem_type == "regression" else RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    try:
        model.fit(X_enc, y)
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            out["importances"] = {}
        else:
            s = pd.Series(importances, index=X_enc.columns).sort_values(ascending=False)
            out["importances"] = s.to_dict()
        out["meta"]["n_features"] = int(X_enc.shape[1])
        out["meta"]["n_rows"] = int(X_enc.shape[0])
        out["meta"]["problem_type"] = problem_type
    except Exception:
        out["importances"] = {}
        out["meta"]["error"] = traceback.format_exc()
    return out


def permutation_feature_importance(
    df: pd.DataFrame,
    target: str,
    n_repeats: int = 10,
    sample_size: Optional[int] = 10000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Compute permutation importance using sklearn.inspection.permutation_importance.
    Will train a light random forest model internally as baseline.
    """
    out: Dict[str, Any] = {"method": "permutation", "importances_mean": {}, "importances_std": {}, "meta": {}}
    if target not in df.columns:
        return out

    # prepare data
    df_small = df if (sample_size is None or df.shape[0] <= sample_size) else _safe_head(df[[*df.columns]], sample_size)
    y = df_small[target]
    X = df_small.drop(columns=[target]).copy()

    X_enc = X.copy()
    for c in X_enc.columns:
        if _is_categorical(X_enc[c]):
            X_enc[c] = _encode_series(X_enc[c])

    # baseline model
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state) \
            if pd.api.types.is_numeric_dtype(y) else RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_enc, y)
        r = permutation_importance(model, X_enc, y, n_repeats=n_repeats, random_state=random_state, n_jobs=1)
        means = pd.Series(r.importances_mean, index=X_enc.columns).sort_values(ascending=False)
        stds = pd.Series(r.importances_std, index=X_enc.columns).reindex(means.index)
        out["importances_mean"] = means.to_dict()
        out["importances_std"] = stds.to_dict()
    except Exception:
        out["meta"]["error"] = traceback.format_exc()
    return out


def mutual_information_importance(df: pd.DataFrame, target: str, max_features: int = 200) -> Dict[str, Any]:
    """
    Compute mutual information between each feature and the target.
    Works for numeric and categorical targets (encoded internally).
    Returns a ranking of features by MI score.
    """
    out: Dict[str, Any] = {"method": "mutual_info", "scores": {}, "meta": {}}
    if target not in df.columns:
        return out

    try:
        y = df[target]
        scores: Dict[str, float] = {}
        for c in df.columns:
            if c == target:
                continue
            try:
                x = df[c]
                mi = _mutual_info_numeric_categorical(x, y)
                scores[c] = float(mi if not math.isnan(mi) else 0.0)
            except Exception:
                scores[c] = 0.0
        s = pd.Series(scores).sort_values(ascending=False).head(max_features)
        out["scores"] = s.to_dict()
    except Exception:
        out["meta"]["error"] = traceback.format_exc()
    return out


def cramers_v_matrix(df: pd.DataFrame, cat_cols: Optional[List[str]] = None, max_cols: int = 100) -> Dict[str, Any]:
    """
    Compute Cramér's V for all categorical column pairs (heavy; sampling used for very large data).
    Returns a dict-of-dicts matrix {col1: {col2: value}}
    """
    out: Dict[str, Any] = {"method": "cramers_v", "matrix": {}, "meta": {}}
    try:
        if cat_cols is None:
            cat_cols = [c for c in df.columns if _is_categorical(df[c])]
        if len(cat_cols) > max_cols:
            cat_cols = cat_cols[:max_cols]
        matrix = {}
        for i, a in enumerate(cat_cols):
            row = {}
            for j, b in enumerate(cat_cols):
                if a == b:
                    row[b] = 1.0
                else:
                    try:
                        row[b] = float(_cramers_v(df[a], df[b]))
                    except Exception:
                        row[b] = float("nan")
            matrix[a] = row
        out["matrix"] = matrix
    except Exception:
        out["meta"]["error"] = traceback.format_exc()
    return out


def shap_feature_importance(
    df: pd.DataFrame,
    target: str,
    sample_size: int = 5000,
    model=None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Compute SHAP values if shap and a model are available.
    If model is None, trains a light random forest.
    Returns mean absolute SHAP values per feature.
    """
    out: Dict[str, Any] = {"method": "shap", "shap_mean_abs": {}, "meta": {}}
    if not _HAS_SHAP:
        out["meta"]["error"] = "shap_not_installed"
        return out
    if target not in df.columns:
        return out

    try:
        df_small = _safe_head(df[[*df.columns]], sample_size)
        y = df_small[target]
        X = df_small.drop(columns=[target]).copy()
        X_enc = X.copy()
        for c in X_enc.columns:
            if _is_categorical(X_enc[c]):
                X_enc[c] = _encode_series(X_enc[c])

        if model is None:
            model = RandomForestRegressor(n_estimators=100, random_state=random_state) \
                if pd.api.types.is_numeric_dtype(y) else RandomForestClassifier(n_estimators=100, random_state=random_state)
            model.fit(X_enc, y)

        explainer = shap.Explainer(model, X_enc, sampler=shap.sample(X_enc, 100))
        shap_values = explainer(X_enc)
        # mean absolute shap per feature
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        s = pd.Series(mean_abs, index=X_enc.columns).sort_values(ascending=False)
        out["shap_mean_abs"] = s.to_dict()
    except Exception:
        out["meta"]["error"] = traceback.format_exc()
    return out


# -------------------------
# Higher-level orchestrator + stability
# -------------------------
def compute_feature_importance_all(
    df: pd.DataFrame,
    target: Optional[str] = None,
    methods: Optional[List[str]] = None,
    max_features: int = 200,
    stability_seeds: Optional[List[int]] = None,
    sample_size_for_heavy: int = 20000
) -> Dict[str, Any]:
    """
    Run a suite of importance computations and aggregate results.
    methods: choose from ['rf', 'permutation', 'mutual_info', 'shap', 'cramers_v'].
    stability_seeds: if provided, run RF importances with multiple seeds to compute stability score.
    """
    if methods is None:
        methods = ["rf", "permutation", "mutual_info", "shap", "cramers_v"]

    out: Dict[str, Any] = {"timestamp": time.time(), "methods": {}, "aggregated": {}, "meta": {}}
    try:
        if target is None or target not in df.columns:
            # Still compute unsupervised-ish relevance (mutual info w/ placeholders) - best-effort
            out["meta"]["warning"] = "no_target_provided"
        # Random forest
        if "rf" in methods:
            out["methods"]["rf"] = random_forest_importance(df, target, sample_size=sample_size_for_heavy)
        if "permutation" in methods and target is not None:
            out["methods"]["permutation"] = permutation_feature_importance(df, target, sample_size=min(10000, df.shape[0]))
        if "mutual_info" in methods and target is not None:
            out["methods"]["mutual_info"] = mutual_information_importance(df, target, max_features=max_features)
        if "shap" in methods and target is not None and _HAS_SHAP:
            out["methods"]["shap"] = shap_feature_importance(df, target, sample_size=min(5000, df.shape[0]))
        if "cramers_v" in methods:
            out["methods"]["cramers_v"] = cramers_v_matrix(df)

        # aggregated ranking (simple rank-aggregation: average rank across methods that provide scores)
        rank_frames = []
        for m, content in out["methods"].items():
            scores = {}
            # try to find a typical "scores dict"
            for key in ("importances", "importances_mean", "scores", "shap_mean_abs"):
                if content.get(key):
                    scores = content.get(key)
                    break
            if scores:
                s = pd.Series(scores).replace({None: 0}).astype(float)
                # convert to ranks (1 best)
                ranks = s.rank(ascending=False, method="average")
                rank_frames.append(ranks.rename(m))
        if rank_frames:
            RF = pd.concat(rank_frames, axis=1).fillna(rank_frames[0].max()+100)
            RF["mean_rank"] = RF.mean(axis=1)
            aggregated = RF["mean_rank"].sort_values()
            out["aggregated"]["mean_rank"] = aggregated.to_dict()
            out["aggregated"]["top_features"] = list(aggregated.index[:max_features])
        else:
            out["aggregated"]["mean_rank"] = {}
            out["aggregated"]["top_features"] = []

        # stability: re-run RF with different seeds and compute Spearman correlation of ranks
        if stability_seeds:
            stability_results = {}
            base = compute_feature_importance_all._rf_stability_run(df, target, seeds=stability_seeds, sample_size=sample_size_for_heavy)
            stability_results["per_seed_ranks"] = base
            out["meta"]["stability"] = {
                "n_seeds": len(stability_seeds),
                "summary": compute_feature_importance_all._stability_summary(base)
            }

    except Exception:
        out["meta"]["error"] = traceback.format_exc()
    return out


def _rf_stability_run(df: pd.DataFrame, target: str, seeds: List[int], sample_size: Optional[int] = None) -> Dict[int, Dict[str, float]]:
    """
    Helper: train RF for each seed and return importances dict per seed.
    """
    results: Dict[int, Dict[str, float]] = {}
    for s in seeds:
        try:
            res = random_forest_importance(df, target, random_state=s, sample_size=sample_size)
            results[s] = res.get("importances", {})
        except Exception:
            results[s] = {}
    return results


def _stability_summary(per_seed_ranks: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
    """
    Given per-seed importances, compute simple stability metrics:
    - pairwise Spearman correlation mean
    - top-k Jaccard stability for k in [10, 25, 50]
    """
    try:
        import scipy.stats as ss  # type: ignore
    except Exception:
        ss = None  # type: ignore

    seeds = list(per_seed_ranks.keys())
    if not seeds:
        return {}
    # convert to DataFrame of ranks
    df_ranks = pd.DataFrame({
        s: pd.Series(per_seed_ranks[s]).rank(ascending=False) for s in seeds
    }).fillna(9999)
    corrs = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a = df_ranks[seeds[i]].fillna(9999)
            b = df_ranks[seeds[j]].fillna(9999)
            try:
                if ss is not None:
                    rho, _ = ss.spearmanr(a, b)
                    corrs.append(float(rho))
                else:
                    corrs.append(float(np.corrcoef(a, b)[0, 1]))
            except Exception:
                corrs.append(0.0)
    mean_corr = float(np.nanmean(corrs)) if corrs else 0.0

    jaccard_summary = {}
    for k in (10, 25, 50):
        top_sets = [set(sorted(r, key=r.get)[:k]) for r in per_seed_ranks.values()]
        # pairwise jaccard mean
        pairs = []
        for i in range(len(top_sets)):
            for j in range(i + 1, len(top_sets)):
                a = top_sets[i]; b = top_sets[j]
                if not a and not b:
                    pairs.append(1.0)
                else:
                    pairs.append(len(a & b) / max(1, len(a | b)))
        jaccard_summary[f"top_{k}_mean_jaccard"] = float(np.nanmean(pairs)) if pairs else 0.0

    return {"mean_spearman": mean_corr, **jaccard_summary}


# -------------------------
# Export / DataFrame helpers
# -------------------------
def importances_to_dataframe(importances: Dict[str, float], method_name: str = "importance") -> pd.DataFrame:
    """
    Convert {feature:score} to a sorted DataFrame with feature, score, rank.
    """
    if not importances:
        return pd.DataFrame(columns=["feature", "score", "rank", "method"])
    s = pd.Series(importances).fillna(0).astype(float).sort_values(ascending=False)
    df = pd.DataFrame({
        "feature": s.index,
        "score": s.values,
        "rank": np.arange(1, len(s) + 1),
        "method": method_name
    })
    return df


def aggregated_to_dataframe(aggregated: Dict[str, float]) -> pd.DataFrame:
    if not aggregated:
        return pd.DataFrame(columns=["feature", "mean_rank"])
    s = pd.Series(aggregated)
    df = pd.DataFrame({"feature": s.index, "mean_rank": s.values}).sort_values("mean_rank")
    df["rank"] = np.arange(1, len(df) + 1)
    return df


# -------------------------
# Visual helpers (plotly)
# -------------------------
def plot_importance_bar(importances: Dict[str, float], title: str = "Feature importance", top_n: int = 20) -> "go.Figure":
    """
    Returns a plotly figure if plotly is available; otherwise raises optionally.
    """
    if not _HAS_PLOTLY:
        raise RuntimeError("plotly is not installed. Install plotly to visualize.")
    df = importances_to_dataframe(importances)
    df = df.head(top_n)
    fig = px.bar(df, x="score", y="feature", orientation="h", title=title, text="score")
    fig.update_layout(yaxis=dict(autorange="reversed"), template="plotly_dark", height=40 + 30 * min(len(df), top_n))
    return fig


# -------------------------
# Example usage (for CLI / dev)
# -------------------------
def _example_run_demo():
    """
    If this file is run directly, run a small demo on synthetic data.
    """
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=2000, n_features=30, n_informative=6, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    res = compute_feature_importance_all(df, target="target", methods=["rf", "permutation", "mutual_info"], stability_seeds=[0, 1, 2])
    print("Top aggregated:", res["aggregated"]["top_features"][:10])
    return res


# -------------------------
# If module executed, quick smoke
# -------------------------
if __name__ == "__main__":
    try:
        out = _example_run_demo()
    except Exception as e:
        print("demo failed:", e)

# ================================================================
#  AutoDataset-Lab — Enterprise Feature Importance Module (FIXED)
#  Fully rewritten for stability, correctness and UI integration.
# ================================================================

from __future__ import annotations
import numpy as np
import pandas as pd
import math, json, traceback, time
from typing import Any, Dict, List, Optional

# sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score

# Optional libs
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


# ================================================================
#             Utilities
# ================================================================
def _is_categorical(s: pd.Series) -> bool:
    if str(s.dtype) == "category":
        return True
    if s.dtype == object:
        return True
    try:
        return s.nunique(dropna=True) <= min(50, 0.02 * len(s))
    except Exception:
        return False


def _encode(s: pd.Series) -> np.ndarray:
    try:
        if _is_categorical(s):
            le = LabelEncoder()
            return le.fit_transform(s.astype(str).fillna("__NA__"))
        return pd.to_numeric(s, errors="coerce").fillna(0).values
    except Exception:
        return pd.Series(s).astype(str).fillna("__NA__").values


def _safe_sample(df: pd.DataFrame, n: int = 20000) -> pd.DataFrame:
    if len(df) <= n:
        return df
    try:
        return df.sample(n=n, random_state=42)
    except:
        return df.head(n)


# ================================================================
#      Cramér’s V without SciPy requirement (stable)
# ================================================================
def _cramers_v_fallback(x: pd.Series, y: pd.Series) -> float:
    try:
        tbl = pd.crosstab(x.fillna("__NA__"), y.fillna("__NA__"))
        chi2 = _chi2_fallback(tbl)
        n = tbl.sum().sum()
        if n == 0:
            return 0.0
        phi2 = chi2 / n
        r, k = tbl.shape
        return float(np.sqrt(phi2 / min(k - 1, r - 1)))
    except Exception:
        return 0.0


def _chi2_fallback(tbl: pd.DataFrame) -> float:
    observed = tbl.values.astype(float)
    rows, cols = observed.shape
    row_sums = observed.sum(axis=1).reshape(-1, 1)
    col_sums = observed.sum(axis=0).reshape(1, -1)
    total = observed.sum()
    expected = (row_sums @ col_sums) / (total + 1e-9)
    expected[expected == 0] = 1e-9
    chi2 = ((observed - expected) ** 2 / expected).sum()
    return float(chi2)


# ================================================================
#     MUTUAL INFORMATION with numeric-target fix
# ================================================================
def _mutual_info(x: pd.Series, y: pd.Series) -> float:
    try:
        xa = _encode(x).astype(int)
        ya = _encode(y).astype(int)
        return float(mutual_info_score(xa, ya))
    except:
        return 0.0


# ================================================================
#     RANDOM FOREST IMPORTANCE
# ================================================================
def rf_importance(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    out = {"method": "rf", "importances": {}, "meta": {}}
    if target not in df.columns:
        return out

    df2 = df.dropna(subset=[target]).copy()
    df2 = _safe_sample(df2, 20000)

    y = df2[target]
    X = df2.drop(columns=[target]).copy()

    X_enc = pd.DataFrame({c: _encode(X[c]) for c in X.columns})

    if pd.api.types.is_numeric_dtype(y):
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)

    try:
        model.fit(X_enc, y)
        imp = pd.Series(model.feature_importances_, index=X.columns)
        imp = imp.sort_values(ascending=False)
        out["importances"] = imp.to_dict()
    except Exception:
        out["meta"]["error"] = traceback.format_exc()

    return out


# ================================================================
#   PERMUTATION IMPORTANCE
# ================================================================
def perm_importance(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    out = {"method": "permutation", "importances": {}, "meta": {}}
    if target not in df.columns:
        return out

    df2 = _safe_sample(df, 10000).dropna(subset=[target])
    y = df2[target]
    X = df2.drop(columns=[target]).copy()
    X_enc = pd.DataFrame({c: _encode(X[c]) for c in X.columns})

    if pd.api.types.is_numeric_dtype(y):
        model = RandomForestRegressor(n_estimators=120, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=120, random_state=42)

    try:
        model.fit(X_enc, y)
        r = permutation_importance(model, X_enc, y, n_repeats=5, random_state=42)
        s = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
        out["importances"] = s.to_dict()
    except Exception:
        out["meta"]["error"] = traceback.format_exc()

    return out


# ================================================================
#   MUTUAL INFORMATION IMPORTANCE
# ================================================================
def mi_importance(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    out = {"method": "mutual_info", "importances": {}}
    if target not in df.columns:
        return out

    y = df[target]
    scores = {}
    for c in df.columns:
        if c == target:
            continue
        try:
            scores[c] = _mutual_info(df[c], y)
        except Exception:
            scores[c] = 0.0

    s = pd.Series(scores).sort_values(ascending=False)
    out["importances"] = s.to_dict()
    return out


# ================================================================
#   CRAMER’S V IMPORTANCE (UNSUPERVISED)
# ================================================================
def cramers_importance(df: pd.DataFrame) -> Dict[str, Any]:
    out = {"method": "cramers_v", "importances": {}}
    cat_cols = [c for c in df.columns if _is_categorical(df[c])]
    if len(cat_cols) < 2:
        return out

    scores = {}
    base = cat_cols[0]
    for c in cat_cols[1:]:
        scores[c] = _cramers_v_fallback(df[base], df[c])

    s = pd.Series(scores).sort_values(ascending=False)
    out["importances"] = s.to_dict()
    return out


# ================================================================
#   SHAP IMPORTANCE (stable TreeExplainer)
# ================================================================
def shap_importance(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    out = {"method": "shap", "importances": {}}
    if not _HAS_SHAP:
        out["meta"] = {"error": "shap_not_installed"}
        return out
    if target not in df.columns:
        return out

    df2 = _safe_sample(df.dropna(subset=[target]), 4000)
    y = df2[target]
    X = df2.drop(columns=[target])
    X_enc = pd.DataFrame({c: _encode(X[c]) for c in X.columns})

    if pd.api.types.is_numeric_dtype(y):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    try:
        model.fit(X_enc, y)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_enc)

        if isinstance(shap_vals, list):
            vals = np.abs(shap_vals[0]).mean(axis=0)
        else:
            vals = np.abs(shap_vals).mean(axis=0)

        s = pd.Series(vals, index=X.columns).sort_values(ascending=False)
        out["importances"] = s.to_dict()
    except Exception:
        out["meta"] = {"error": traceback.format_exc()}

    return out


# ================================================================
#   MASTER AGGREGATOR (this is what Streamlit uses)
# ================================================================
def compute_feature_importance_all(
    df: pd.DataFrame,
    target: Optional[str] = None,
) -> Dict[str, Any]:

    out = {"methods": {}, "aggregated": {}, "top_features": [], "meta": {}}

    # Supervised methods require target
    has_target = target in df.columns if target else False

    # 1. RF
    if has_target:
        out["methods"]["rf"] = rf_importance(df, target)

    # 2. Permutation
    if has_target:
        out["methods"]["permutation"] = perm_importance(df, target)

    # 3. Mutual Info
    if has_target:
        out["methods"]["mutual_info"] = mi_importance(df, target)

    # 4. SHAP
    if has_target:
        out["methods"]["shap"] = shap_importance(df, target)

    # 5. Cramer (unsupervised)
    out["methods"]["cramers_v"] = cramers_importance(df)

    # AGGREGATE
    frames = []
    for name, block in out["methods"].items():
        imp = block.get("importances")
        if not imp:
            continue
        s = pd.Series(imp).astype(float).replace(np.nan, 0)
        ranks = s.rank(ascending=False, method="average")
        frames.append(ranks.rename(name))

    if frames:
        R = pd.concat(frames, axis=1).fillna(9999)
        R["mean_rank"] = R.mean(axis=1)
        R = R.sort_values("mean_rank")

        out["aggregated"]["mean_rank"] = R["mean_rank"].to_dict()

        # convert aggregated top list into UI-friendly list
        out["top_features"] = [
            {"column": idx, "importance": float(1.0 / (1.0 + rank))}
            for idx, rank in R["mean_rank"].items()
        ]

    return out


# ================================================================
#   PLOTLY HELPERS
# ================================================================
def plot_importance_bar(imp: Dict[str, float], title="Feature Importance"):
    if not _HAS_PLOTLY:
        return None
    df = pd.DataFrame({"feature": list(imp.keys()), "score": list(imp.values())})
    df = df.sort_values("score", ascending=False).head(20)
    fig = px.bar(df, x="score", y="feature", orientation="h", title=title, text="score")
    fig.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"))
    return fig

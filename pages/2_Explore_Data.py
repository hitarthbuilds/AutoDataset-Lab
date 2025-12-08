# pages/2_Explore_Data.py
import streamlit as st
import pandas as pd
import tempfile
import os
import io
import time
import inspect
import math
from typing import Any, Dict, List, Tuple, Optional

# ---------------------------
# Defensive imports from core (DO NOT CHANGE)
# ---------------------------
try:
    from core.eda.analyze import analyze_dataframe as analyze_df
except Exception:
    analyze_df = None

try:
    from core.eda.missing import summarize_missingness
except Exception:
    summarize_missingness = None

try:
    from core.eda.quality import compute_data_quality_report
except Exception:
    compute_data_quality_report = None

try:
    from core.eda.anamolies import detect_anomalies
except Exception:
    detect_anomalies = None

try:
    from core.eda.drift import detect_dataset_drift
except Exception:
    detect_dataset_drift = None

try:
    from core.eda.feature_importance import compute_feature_importance_all
except Exception:
    compute_feature_importance_all = None

# visualization bundle
try:
    from core.eda.visualize import generate_visual_bundle
except Exception:
    generate_visual_bundle = None

# report export
try:
    from core.eda.report import export_audit_report
except Exception:
    export_audit_report = None

# ---------------------------
# Session initialization
# ---------------------------
_default_session = {
    "df": None,
    "reference_df": None,
    "eda_done": False,
    "analysis": {},
    "missing": {},
    "quality": {},
    "anomalies": {},
    "drift": {},
    "schema": {},
    "feature_importance": {},
    "recommendations": [],
    "visuals": {},
    "last_report_paths": {},
}

for k, v in _default_session.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------
# Small helpers
# ---------------------------
def _safe_dict(d: Any) -> Dict:
    return d if isinstance(d, dict) else {}


def _safe_list(v: Any) -> List:
    return v if isinstance(v, list) else []


def _try_call_export_report(*args, **kwargs):
    """
    Try to call export_audit_report with dynamic arg sets so we handle different signatures
    across environments. Return dict result or {"error": str(e)}.
    """
    if not export_audit_report:
        return {"error": "export_audit_report not available in this environment."}

    try:
        sig = inspect.signature(export_audit_report)
        params = list(sig.parameters.keys())
    except Exception:
        params = []

    # build candidate arg orders (attempts)
    candidates = [
        ("output_path", "output_title", "analysis", "quality", "missing", "anomalies", "drift", "schema", "feature_importance", "visuals", "generate_pdf"),
        ("analysis", "missing", "quality", "anomalies", "drift", "schema", "feature_importance", "visuals"),
        ("analysis", "missing", "quality", "anomalies", "drift", "schema", "feature_importance"),
        ("analysis", "missing", "quality", "anomalies", "drift"),
        ("analysis", "missing"),
    ]

    # try sensible defaults: create temp base path
    tmp_base = os.path.join(tempfile.gettempdir(), f"audit_report_{int(time.time())}")
    # if function expects full signature, try to pass realistic values
    for cand in candidates:
        try_args = []
        for name in cand:
            if name == "analysis":
                try_args.append(st.session_state.get("analysis", {}))
            elif name == "missing":
                try_args.append(st.session_state.get("missing", {}))
            elif name == "quality":
                try_args.append(st.session_state.get("quality", {}))
            elif name == "anomalies":
                try_args.append(st.session_state.get("anomalies", {}))
            elif name == "drift":
                try_args.append(st.session_state.get("drift", {}))
            elif name == "schema":
                try_args.append(st.session_state.get("schema", {}))
            elif name == "feature_importance":
                try_args.append(st.session_state.get("feature_importance", {}))
            elif name == "visuals":
                try_args.append(st.session_state.get("visuals", {}))
            elif name == "output_path":
                try_args.append(tmp_base)
            elif name == "output_title":
                try_args.append("Audit Report - AutoDataset-Lab")
            elif name == "generate_pdf":
                try_args.append(True)
            else:
                # fallback empty
                try_args.append(None)
        res = export_audit_report(*try_args)
        return res if isinstance(res, dict) else {"result": res}
    # last resort
    try:
        res = export_audit_report()
        return res if isinstance(res, dict) else {"result": res}
    except Exception as e:
        return {"error": f"export_audit_report failed: {e}"}


# ---------------------------
# Formatting / humanization helpers
# ---------------------------
def _missing_to_human(missing: Dict[str, Any]) -> Tuple[List[str], pd.DataFrame]:
    """
    Convert missingness structure into a list of human sentences and a DataFrame summary.
    Handles multiple shapes defensively so missing keys never crash.
    """
    out = []
    missing = _safe_dict(missing)

    # Try all common shapes used by your modules
    per_col = (
        _safe_dict(missing.get("per_column"))
        or _safe_dict(missing.get("columns"))
        or _safe_dict(missing.get("perColumn"))
        or {}
    )

    rows = []

    # Build rows safely
    for col, info in per_col.items():
        missing_count = 0
        missing_pct = 0.0

        if isinstance(info, dict):
            # try all possible key names
            missing_count = info.get("missing")
            if missing_count is None:
                missing_count = info.get("missing_count", info.get("count", 0))

            missing_pct = info.get("missing_percent")
            if missing_pct is None:
                missing_pct = info.get("missing_pct", info.get("pct", 0))

        elif isinstance(info, (int, float)):
            # scalar means count only
            missing_count = int(info)
            missing_pct = 0.0

        rows.append({
            "column": col,
            "missing": int(missing_count or 0),
            "missing_percent": float(round(missing_pct or 0, 2)),
        })

    # Always produce a DataFrame with correct columns
    df = pd.DataFrame(rows, columns=["column", "missing", "missing_percent"])

    # If no rows or zero-missing everywhere
    if df.empty or df["missing"].sum() == 0:
        out.append("No missing values detected.")
        return out, df

    # Safe sort (column now always exists)
    df = df.sort_values("missing", ascending=False)

    out.append(f"Total missing values across dataset: {int(df['missing'].sum())}.")
    out.append("Top columns by missingness:")

    for _, r in df.head(5).iterrows():
        out.append(f"- `{r['column']}` → {int(r['missing'])} missing ({r['missing_percent']}%)")

    return out, df


def _quality_to_human(quality: Dict[str, Any]) -> Tuple[List[str], pd.DataFrame]:
    out = []
    quality = _safe_dict(quality)
    if not quality:
        out.append("No quality issues detected or quality analyzer unavailable.")
        return out, pd.DataFrame()

    # try a number of common keys
    dup = quality.get("duplicate_summary") or quality.get("duplicates") or {}
    dup_count = dup.get("duplicate_count") or dup.get("duplicate_rows") or dup.get("duplicates", 0)
    if dup_count:
        out.append(f"Duplicate rows: {int(dup_count)}.")

    const = quality.get("constant_summary", {}) or {}
    const_cols = const.get("constant_columns") or const.get("constants") or []
    if const_cols:
        out.append(f"Constant columns detected: {len(const_cols)}.")
        for item in const_cols[:5]:
            col = item.get("column") if isinstance(item, dict) else str(item)
            out.append(f"- `{col}`")

    stats_warn = quality.get("warnings") or quality.get("quality_warnings") or []
    if stats_warn:
        out.append("Quality warnings:")
        for w in stats_warn[:8]:
            out.append(f"- {w}")

    # build a small table
    rows = [
        {"metric": "duplicate_rows", "value": int(dup_count or 0)},
        {"metric": "constant_cols", "value": len(const_cols)},
    ]
    dfq = pd.DataFrame(rows)
    return out, dfq


def _anomalies_to_human(anom: Dict[str, Any]) -> List[str]:
    out = []
    anom = _safe_dict(anom)
    methods = _safe_dict(anom.get("methods", {}))
    if not methods:
        out.append("No anomalies detected or anomaly module unavailable.")
        return out
    out.append("Anomaly detection summary:")
    for m, detail in methods.items():
        per = _safe_dict(detail.get("per_column", {}))
        if per:
            try:
                total = sum(int(v if isinstance(v, (int, float)) else v.get("count", 0)) for v in per.values())
            except Exception:
                total = 0
            out.append(f"- `{m}` flagged approximately {total} anomalies across columns.")
        else:
            out.append(f"- `{m}` ran but no per-column counts available.")
    return out


def _drift_to_human(drift: Dict[str, Any]) -> List[str]:
    out = []
    drift = _safe_dict(drift)
    if not drift:
        out.append("No dataset drift detected or drift module unavailable.")
        return out
    by_col = _safe_dict(drift.get("per_column", {}) or drift.get("drift_by_column", {}) or drift.get("columns", {}))
    flagged = []
    for c, info in by_col.items():
        try:
            if isinstance(info, dict) and (info.get("drift_detected") or info.get("drift", False) or info.get("p_value", 0) < 0.05):
                flagged.append(c)
        except Exception:
            continue
    out.append(f"Columns suspected of distributional drift: {len(flagged)}.")
    for c in flagged[:10]:
        out.append(f"- `{c}`")
    return out


# ---------------------------
# Feature importance aggregation (Mode 2: average normalized method scores)
# ---------------------------
def _collect_method_scores(fi: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Given compute_feature_importance_all output, extract per-method feature->score dicts.
    Return dict: { method_name: {feature: score, ...}, ... }
    Defensive: handles many shapes.
    """
    res = {}
    fi = _safe_dict(fi)
    methods = fi.get("methods") or {}
    # If top-level flattened scores exist, capture them
    # e.g., fi.get("scores") or fi.get("importances")
    top_level_candidates = ("importances", "importances_mean", "scores", "shap_mean_abs")
    for cand in top_level_candidates:
        if cand in fi and isinstance(fi[cand], dict):
            res[f"top_{cand}"] = {k: float(v) for k, v in fi[cand].items()}

    if isinstance(methods, dict):
        for m, detail in methods.items():
            detail = _safe_dict(detail)
            # check many keys
            for key in ("importances", "importances_mean", "scores", "shap_mean_abs"):
                dd = detail.get(key)
                if isinstance(dd, dict):
                    # convert numeric-like to floats
                    try:
                        res[f"{m}.{key}"] = {k: float(v) for k, v in dd.items()}
                        break
                    except Exception:
                        # fallback: try pandas-friendly conversion
                        res[f"{m}.{key}"] = {k: float(dd.get(k) or 0.0) for k in dd.keys()}
                        break
            # some methods might store as list of dicts [{feature, score}, ...]
            if not any(k.startswith(f"{m}.") for k in res.keys()):
                # try list shapes
                for possible in ("ranked", "top_features", "scores_list", "features"):
                    lst = detail.get(possible)
                    if isinstance(lst, list):
                        mapping = {}
                        for item in lst:
                            if isinstance(item, dict):
                                keyf = item.get("feature") or item.get("column") or item.get("name")
                                val = item.get("score") or item.get("importance") or item.get("value")
                                if keyf is not None:
                                    try:
                                        mapping[str(keyf)] = float(val or 0.0)
                                    except Exception:
                                        mapping[str(keyf)] = 0.0
                        if mapping:
                            res[f"{m}.{possible}"] = mapping
                            break
    return res


def _normalize_series_dict(d: Dict[str, float]) -> Dict[str, float]:
    """
    Min-max normalize a dict of numeric values to 0..1 (higher is better).
    If all values equal, map top to 1.0 and others to 0.
    """
    if not d:
        return {}
    try:
        s = pd.Series(d).astype(float)
    except Exception:
        # fallback to manual
        out = {}
        for k, v in d.items():
            try:
                out[k] = float(v)
            except Exception:
                out[k] = 0.0
        s = pd.Series(out)
    mn = s.min()
    mx = s.max()
    if math.isclose(mx, mn):
        # all same -> top gets 1.0, rest 0 (or all 0 if zero)
        if mx == 0:
            return {k: 0.0 for k in s.index}
        out = {}
        top = s.idxmax()
        for k in s.index:
            out[k] = 1.0 if k == top else 0.0
        return out
    norm = (s - mn) / (mx - mn)
    return norm.to_dict()


def _aggregate_normalized_methods(method_score_maps: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    For each method map, normalize then average across methods.
    Returns aggregated_score: {feature: score}
    """
    if not method_score_maps:
        return {}
    normalized_maps = []
    for mname, fmap in method_score_maps.items():
        # cramers_v might be matrix; skip non-1d
        if not isinstance(fmap, dict):
            continue
        normalized = _normalize_series_dict(fmap)
        normalized_maps.append(normalized)

    # collect union of features
    all_feats = set()
    for nm in normalized_maps:
        all_feats.update(nm.keys())
    if not all_feats:
        return {}

    # build dataframe: rows=feature, cols=method -> fill 0 for missing
    dfm = pd.DataFrame(index=sorted(all_feats))
    for i, nm in enumerate(normalized_maps):
        dfm[f"m{i}"] = pd.Series(nm)
    dfm = dfm.fillna(0.0)
    # average across columns
    dfm["mean_score"] = dfm.mean(axis=1)
    # convert to dict
    return dfm["mean_score"].to_dict()


def _compute_aggregated_feature_importance(fi_raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce canonical aggregated feature_importance dict that downstream UI expects.
    Structure:
    {
      "methods": {...},            # original raw methods (kept)
      "aggregated_scores": {feat: score (0..1)},
      "top_features": [{"column": feat, "importance": score}, ...],
      "meta": {...}
    }
    """
    out = {"methods": {}, "aggregated_scores": {}, "top_features": [], "meta": {}}
    if not fi_raw:
        return out
    try:
        # keep raw
        out["methods"] = fi_raw.get("methods", {})
        # collect per-method maps
        method_maps = _collect_method_scores(fi_raw)
        if not method_maps:
            # fallback: if fi_raw is itself mapping col->score
            if isinstance(fi_raw, dict) and all(isinstance(v, (int, float)) for v in fi_raw.values()):
                ag = _normalize_series_dict({k: float(v) for k, v in fi_raw.items()})
                out["aggregated_scores"] = ag
                out["top_features"] = [{"column": k, "importance": float(v)} for k, v in sorted(ag.items(), key=lambda x: x[1], reverse=True)]
                return out
            return out

        aggregated = _aggregate_normalized_methods(method_maps)
        # final normalization (min-max again to 0..1)
        aggregated = _normalize_series_dict(aggregated)
        out["aggregated_scores"] = aggregated
        # top features
        ordered = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
        out["top_features"] = [{"column": k, "importance": float(v)} for k, v in ordered]
        return out
    except Exception as e:
        out["meta"]["error"] = str(e)
        return out


# ---------------------------
# Main UI
# ---------------------------
st.set_page_config(layout="wide")
st.title("🔍 Explore Data — AutoDataset-Lab  (Mode 2 FI aggregation)")

df = st.session_state.get("df")
reference_df = st.session_state.get("reference_df")

if df is None or not isinstance(df, pd.DataFrame):
    st.warning("Upload a dataset first (go to Upload Dataset).")
    st.stop()

# Sidebar
st.sidebar.header("Processing Options")
max_rows = st.sidebar.number_input("Sample rows for heavy ops", min_value=100, max_value=50000, value=5000)
target_col = st.sidebar.text_input("Target column (optional)", "")
ref_file = st.sidebar.file_uploader("Reference CSV (optional, for drift)", type=["csv"])
if ref_file:
    try:
        st.session_state["reference_df"] = pd.read_csv(ref_file)
        reference_df = st.session_state["reference_df"]
    except Exception:
        st.sidebar.error("Failed to load reference dataset.")
generate_report_flag = st.sidebar.checkbox("Enable HTML/PDF report generation", value=True)

if st.sidebar.button("Run EDA"):
    # reset
    for k in ["eda_done", "analysis", "missing", "quality", "anomalies", "drift", "schema", "feature_importance", "visuals", "recommendations", "last_report_paths"]:
        st.session_state[k] = {} if k != "eda_done" else False

    with st.spinner("Running EDA pipeline (defensive mode)… this may take a while on large datasets"):
        # sample
        sample_n = min(int(max_rows), len(df))
        df_sample = df.sample(n=sample_n, random_state=42) if sample_n < len(df) else df.copy()

        # 1) Analysis
        if analyze_df:
            try:
                analysis = analyze_df(df_sample)
                st.session_state["analysis"] = analysis or {}
                # build a lightweight schema for UI
                try:
                    cols_over = analysis.get("columns_overview", {}) or {}
                    schema = {}
                    for c in df_sample.columns:
                        colinfo = cols_over.get(c) or {}
                        dtype = colinfo.get("dtype") if isinstance(colinfo, dict) else str(df_sample[c].dtype)
                        inferred = None
                        sem = colinfo.get("semantic") if isinstance(colinfo, dict) else None
                        if isinstance(sem, dict):
                            inferred = sem.get("inferred_type")
                        schema[c] = {"dtype": dtype, "inferred_type": inferred}
                    st.session_state["schema"] = schema
                except Exception:
                    st.session_state["schema"] = {}
            except Exception:
                st.session_state["analysis"] = {}
        else:
            st.session_state["analysis"] = {"rows": len(df_sample), "columns": len(df_sample.columns), "columns_overview": {}}
            st.session_state["schema"] = {c: {"dtype": str(df_sample[c].dtype), "inferred_type": None} for c in df_sample.columns}

        # 2) Missingness
        if summarize_missingness:
            try:
                st.session_state["missing"] = summarize_missingness(df_sample) or {}
            except Exception:
                st.session_state["missing"] = {}
        else:
            # fallback summary
            per = {}
            for c in df_sample.columns:
                miss = int(df_sample[c].isna().sum())
                pct = round(100 * miss / len(df_sample), 2) if len(df_sample) else 0.0
                per[c] = {"missing": miss, "missing_percent": pct}
            st.session_state["missing"] = {"per_column": per}

        # 3) Quality
        if compute_data_quality_report:
            try:
                st.session_state["quality"] = compute_data_quality_report(df_sample) or {}
            except Exception:
                st.session_state["quality"] = {}
        else:
            st.session_state["quality"] = {}

        # 4) Anomalies
        if detect_anomalies:
            try:
                st.session_state["anomalies"] = detect_anomalies(df_sample) or {}
            except Exception:
                st.session_state["anomalies"] = {}
        else:
            st.session_state["anomalies"] = {}

        # 5) Drift
        if detect_dataset_drift and isinstance(reference_df, pd.DataFrame):
            try:
                st.session_state["drift"] = detect_dataset_drift(df_sample, reference_df) or {}
            except Exception:
                st.session_state["drift"] = {}
        else:
            st.session_state["drift"] = {}

        # 6) Feature importance (compute and then aggregate Mode 2)
        fi_raw = {}
        if compute_feature_importance_all:
            try:
                fi_raw = compute_feature_importance_all(df_sample, target_col if target_col else None) or {}
            except Exception:
                fi_raw = {}
        else:
            fi_raw = {}
        fi_agg = _compute_aggregated_feature_importance(fi_raw)
        # store both raw + aggregated for UI/reporting
        st.session_state["feature_importance"] = {"raw": fi_raw, "aggregated": fi_agg}

        # 7) Visuals (do not modify visualize internals; just handle returns safely)
        if generate_visual_bundle:
            try:
                visuals = generate_visual_bundle(df_sample, st.session_state.get("missing", {}), st.session_state.get("quality", {}), st.session_state.get("anomalies", {}), st.session_state.get("drift", {}), st.session_state.get("feature_importance", {}))
                # normalize shapes
                if isinstance(visuals, dict):
                    st.session_state["visuals"] = visuals
                elif isinstance(visuals, list):
                    vdict = {}
                    for i, item in enumerate(visuals):
                        if isinstance(item, tuple) and len(item) == 2:
                            title, fig = item
                        else:
                            title, fig = f"visual_{i}", item
                        vdict[str(title)] = fig
                    st.session_state["visuals"] = vdict
                else:
                    # single figure object
                    st.session_state["visuals"] = {"visual_0": visuals}
            except Exception as e:
                st.session_state["visuals"] = {"__error__": f"generate_visual_bundle failed: {e}"}
        else:
            st.session_state["visuals"] = {}

        # 8) Human recommendations (actionable)
        # reuse helper that focuses on actionable items
        try:
            recs = []
            # missing
            miss_lines, miss_df = _missing_to_human(st.session_state.get("missing", {}))
            recs.extend(miss_lines[:5])
            # quality
            q_lines, qdf = _quality_to_human(st.session_state.get("quality", {}))
            recs.extend(q_lines[:5])
            # anomalies
            recs.extend(_anomalies_to_human(st.session_state.get("anomalies", {}))[:5])
            # drift
            recs.extend(_drift_to_human(st.session_state.get("drift", {}))[:5])
            # feature importance quick note
            topf = fi_agg.get("top_features", [])[:5]
            if topf:
                recs.append("Top predictive features (aggregated): " + ", ".join([f"`{t['column']}`" for t in topf]))
            if not recs:
                recs = ["No specific recommendations generated."]
            st.session_state["recommendations"] = recs
        except Exception:
            st.session_state["recommendations"] = ["Failed to synthesize recommendations."]

        # 9) Report generation (best-effort)
        if generate_report_flag:
            paths = _try_call_export_report(
                st.session_state.get("analysis", {}),
                st.session_state.get("missing", {}),
                st.session_state.get("quality", {}),
                st.session_state.get("anomalies", {}),
                st.session_state.get("drift", {}),
                st.session_state.get("schema", {}),
                st.session_state.get("feature_importance", {}),
                st.session_state.get("visuals", {}),
            )
            st.session_state["last_report_paths"] = paths

    st.session_state["eda_done"] = True
    st.success("EDA complete. Scroll tabs for details.")


# ---------------------------
# Metrics row (compact)
# ---------------------------
analysis = _safe_dict(st.session_state.get("analysis", {}))
rows = int(analysis.get("rows", len(df)))
cols = int(analysis.get("columns", len(df.columns)))

# missing total computed defensively
missing_total = 0
try:
    missing_struct = _safe_dict(st.session_state.get("missing", {}))
    percol = _safe_dict(missing_struct.get("per_column", {}))
    for v in percol.values():
        if isinstance(v, dict):
            missing_total += int(v.get("missing", v.get("missing_count", 0)))
        elif isinstance(v, (int, float)):
            missing_total += int(v)
except Exception:
    missing_total = 0

st.markdown(f"""
### Rows: **{rows}**    &nbsp;&nbsp;&nbsp; Columns: **{cols}**    &nbsp;&nbsp;&nbsp; Missing values: **{missing_total}**    &nbsp;&nbsp;&nbsp; Duplicate rows: **0**
""")


# Tabs
tabs = st.tabs(["Overview", "Schema", "Missing", "Anomalies", "Drift", "Quality", "Feature Importances", "Recommendations", "Report", "Visuals"])

# -------------
# OVERVIEW (Executive)
# -------------
with tabs[0]:
    st.subheader("Executive Summary")
    if not st.session_state["eda_done"]:
        st.info("EDA not run yet. Click 'Run EDA' to compute analyses.")
    else:
        exec_lines = []
        exec_lines.append(f"Dataset: **{rows} rows** × **{cols} columns**.")
        # missing
        miss_lines, miss_df = _missing_to_human(st.session_state.get("missing", {}))
        if miss_df is not None and not miss_df.empty:
            total_miss = int(miss_df["missing"].sum())
            exec_lines.append(f"Total missing values: **{total_miss}**. Top missing columns: " + ", ".join([f"`{c}`" for c in miss_df.head(3)["column"].tolist()]) + ".")
        else:
            exec_lines.append("No missing values detected.")
        # quality
        q_lines, qdf = _quality_to_human(st.session_state.get("quality", {}))
        if qdf is not None and not qdf.empty:
            exec_lines.append("Data quality issues detected (see Quality tab).")
        else:
            exec_lines.append("No major data quality issues detected.")
        # anomalies + drift
        anom_lines = _anomalies_to_human(st.session_state.get("anomalies", {}))
        drift_lines = _drift_to_human(st.session_state.get("drift", {}))

        # show bullets succinctly
        for l in exec_lines:
            st.markdown(f"- {l}")
        if anom_lines:
            st.markdown("**Anomalies (summary):**")
            for l in anom_lines[:4]:
                st.markdown(f"- {l}")
        if drift_lines:
            st.markdown("**Drift (summary):**")
            for l in drift_lines[:4]:
                st.markdown(f"- {l}")

# -------------
# SCHEMA TAB
# -------------
with tabs[1]:
    schema = _safe_dict(st.session_state.get("schema", {}))
    if not schema:
        st.info("No schema available.")
    else:
        df_schema = pd.DataFrame([{"column": col, "dtype": info.get("dtype"), "semantic": info.get("inferred_type")} for col, info in schema.items()])
        st.dataframe(df_schema)

# -------------
# MISSING TAB
# -------------
with tabs[2]:
    st.subheader("Missingness Overview")
    miss_lines, df_missing = _missing_to_human(st.session_state.get("missing", {}))
    for l in miss_lines:
        st.markdown(f"- {l}")
    if df_missing is not None and not df_missing.empty:
        st.markdown("### Missingness table (top 200)")
        st.dataframe(df_missing.head(200))

# -------------
# ANOMALIES TAB
# -------------
with tabs[3]:
    st.subheader("Detected Anomalies")
    anomalies = _safe_dict(st.session_state.get("anomalies", {}))
    if not anomalies:
        st.info("No anomalies detected or module unavailable.")
    else:
        anom_lines = _anomalies_to_human(anomalies)
        for l in anom_lines:
            st.markdown(f"- {l}")
        methods = _safe_dict(anomalies.get("methods", {}))
        for method_name, detail in methods.items():
            st.markdown(f"### Method: {method_name}")
            per_col = _safe_dict(detail.get("per_column", {}))
            rows = []
            for c, info in per_col.items():
                if isinstance(info, dict):
                    count = info.get("count", info.get("anomaly_count", 0))
                elif isinstance(info, (int, float)):
                    count = int(info)
                else:
                    count = 0
                rows.append({"column": c, "anomaly_count": int(count)})
            if rows:
                st.dataframe(pd.DataFrame(rows).sort_values("anomaly_count", ascending=False))

# -------------
# DRIFT TAB
# -------------
with tabs[4]:
    st.subheader("Dataset Drift")
    drift = _safe_dict(st.session_state.get("drift", {}))
    for l in _drift_to_human(drift):
        st.markdown(f"- {l}")
    if drift:
        with st.expander("See raw drift JSON"):
            st.json(drift)

# -------------
# QUALITY TAB
# -------------
with tabs[5]:
    st.subheader("Data Quality")

    quality = _safe_dict(st.session_state.get("quality", {}))
    q_lines, qdf = _quality_to_human(quality)

    # Render human-readable lines
    for l in q_lines:
        st.markdown(f"- {l}")

    # --- Build Enhanced Quality Metrics ---
    col_count = len(df.columns)
    row_count = len(df)

    # Missing %
    missing_struct = _safe_dict(st.session_state.get("missing", {}))
    percol = _safe_dict(missing_struct.get("per_column", {}))
    total_missing = sum([v.get("missing", 0) for v in percol.values()]) if percol else 0
    missing_pct = round((total_missing / (row_count * col_count)) * 100, 2) if row_count and col_count else 0

    # Duplicate %
    dup_count = quality.get("duplicate_summary", {}).get("duplicate_count", 0)
    dup_pct = round((dup_count / row_count) * 100, 2) if row_count else 0

    # Constant columns %
    const_cols = quality.get("constant_summary", {}).get("constant_columns", []) or []
    const_pct = round((len(const_cols) / col_count) * 100, 2) if col_count else 0

    # Infinite values
    inf_cols = quality.get("stats_issues", {}).get("infinite_columns", []) if isinstance(quality.get("stats_issues"), dict) else []
    inf_pct = round((len(inf_cols) / col_count) * 100, 2) if col_count else 0

    # Build enhanced table
    enhanced_rows = [
        {"metric": "missing_percent", "value": f"{missing_pct}%"},
        {"metric": "duplicate_percent", "value": f"{dup_pct}%"},
        {"metric": "constant_columns_percent", "value": f"{const_pct}%"},
        {"metric": "infinite_columns_percent", "value": f"{inf_pct}%"},
        {"metric": "duplicate_rows", "value": dup_count},
        {"metric": "constant_cols", "value": len(const_cols)},
        {"metric": "infinite_cols", "value": len(inf_cols)},
    ]

    # Compute overall quality score
    # Simple additive penalty system
    quality_score = 100
    quality_score -= missing_pct * 0.4
    quality_score -= dup_pct * 1.0
    quality_score -= const_pct * 2.5
    quality_score -= inf_pct * 3.0

    quality_score = max(0, min(100, round(quality_score, 2)))

    st.markdown(f"### Overall Data Quality Score: **{quality_score}/100**")

    st.markdown("### Quality summary table")
    st.dataframe(pd.DataFrame(enhanced_rows))


# -------------
# FEATURE IMPORTANCES TAB
# -------------
with tabs[6]:
    st.subheader("Feature Importances (aggregated)")
    fi_state = _safe_dict(st.session_state.get("feature_importance", {}))
    fi_agg = _safe_dict(fi_state.get("aggregated", {}))
    # if previously stored aggregated (compatible older shapes)
    if fi_agg and fi_agg.get("aggregated_scores"):
        ag_scores = fi_agg.get("aggregated_scores", {})
        df_ag = pd.DataFrame([{"column": c, "importance": float(v)} for c, v in ag_scores.items()]).sort_values("importance", ascending=False)
        st.markdown("Top aggregated features:")
        for _, r in df_ag.head(10).iterrows():
            st.markdown(f"- `{r['column']}` → {r['importance']:.4f}")
        st.markdown("### Aggregated feature importance table (top 200)")
        st.dataframe(df_ag.head(200))
    else:
        # try older raw shape
        raw = _safe_dict(fi_state.get("raw", {}))
        # if raw aggregated already present
        if raw and raw.get("aggregated"):
            ag = raw.get("aggregated")
            if isinstance(ag, dict) and ag.get("mean_rank"):
                st.markdown("Feature importance exists (rank aggregation). See raw for details.")
                with st.expander("Raw feature_importance"):
                    st.json(raw)
            else:
                st.info("No usable feature importance found.")
        else:
            st.info("No feature importances available.")

# -------------
# RECOMMENDATIONS TAB (actionable)
# -------------
with tabs[7]:
    st.subheader("Recommendations (actionable)")
    recs = st.session_state.get("recommendations", [])
    if not recs:
        st.info("No recommendations generated.")
    else:
        for r in recs:
            st.markdown(f"- {r}")

# -------------
# REPORT TAB
# -------------
with tabs[8]:
    st.subheader("Last export results")
    paths = _safe_dict(st.session_state.get("last_report_paths", {}))
    if not paths:
        st.info("No report generated yet.")
    else:
        if paths.get("error"):
            st.error(paths.get("error"))
        else:
            st.json(paths)
            html_path = paths.get("html") or paths.get("report_html") or None
            if html_path and isinstance(html_path, str) and os.path.exists(html_path):
                try:
                    with open(html_path, "rb") as f:
                        st.download_button("Download HTML Report", f, file_name="audit_report.html")
                except Exception as e:
                    st.error(f"Failed to open HTML: {e}")

# -------------
# VISUALS TAB
# -------------
with tabs[9]:
    st.subheader("Visual panels")
    visuals = st.session_state.get("visuals", {})
    if not visuals:
        st.info("No visuals produced. Ensure `core.eda.visualize.generate_visual_bundle` exists and returns fig objects or mapping.")
    else:
        # normalize to iterable of (title, fig)
        items = []
        if isinstance(visuals, dict):
            items = list(visuals.items())
        elif isinstance(visuals, list):
            for i, it in enumerate(visuals):
                if isinstance(it, tuple) and len(it) == 2:
                    items.append((str(it[0]), it[1]))
                else:
                    items.append((f"visual_{i}", it))
        else:
            items = [("visual", visuals)]

        any_rendered = False
        for title, fig in items:
            if title == "__error__":
                st.warning(fig)
                continue
            st.markdown(f"### {title}")
            rendered = False
            # try plotly
            try:
                import plotly.graph_objs as _pg
                if "plotly" in str(type(fig)).lower() or hasattr(fig, "to_plotly_json") or hasattr(fig, "data"):
                    st.plotly_chart(fig, use_container_width=True)
                    rendered = True
            except Exception:
                rendered = False
            if rendered:
                any_rendered = True
                continue
            # try matplotlib
            try:
                import matplotlib.pyplot as _plt
                if hasattr(fig, "figure") or hasattr(fig, "savefig") or "matplotlib" in str(type(fig)).lower():
                    st.pyplot(fig)
                    rendered = True
            except Exception:
                rendered = False
            if rendered:
                any_rendered = True
                continue
            # dataframe
            if isinstance(fig, pd.DataFrame):
                st.dataframe(fig)
                any_rendered = True
                continue
            # PIL image
            try:
                from PIL import Image as _Image
                if isinstance(fig, _Image.Image):
                    st.image(fig)
                    any_rendered = True
                    continue
            except Exception:
                pass
            # fallback
            st.write("Cannot render visual object of type:", type(fig))
            try:
                st.write(fig)
            except Exception:
                pass

        if not any_rendered:
            st.info("No visual objects were renderable. Check the return type of generate_visual_bundle.")

# End of file

# pages/2_Explore_Data.py
import streamlit as st
import pandas as pd
import tempfile
import os
import io
import time
import inspect
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
# Config
# ---------------------------
DEBUG = False  # flip to True during development if you want console prints

# ---------------------------
# Small helpers
# ---------------------------
def _safe_dict(d: Any) -> Dict:
    return d if isinstance(d, dict) else {}


def _safe_list(v: Any) -> List:
    return v if isinstance(v, list) else []


def _call_export_report_flexible(
    title: str,
    analysis: Dict[str, Any],
    missing: Dict[str, Any],
    quality: Dict[str, Any],
    anomalies: Dict[str, Any],
    drift: Dict[str, Any],
    schema: Dict[str, Any],
    feature_importance: Dict[str, Any],
    visuals: Optional[Dict[str, Any]] = None,
    generate_pdf: bool = True,
    per_column_limit: int = 6,
) -> Dict[str, Any]:
    """
    Call export_audit_report in a robust way to handle multiple signatures:
    - old: export_audit_report(output_path, output_title, analysis, quality, missing, anomalies, drift, schema, feature_importance, visuals=..., generate_pdf=...)
    - new: export_audit_report(analysis, missing, quality, anomalies, drift, schema, feature_importance, visuals=..., generate_pdf=...)
    - fallback: try keyword-args
    Returns a dict with results or {'error': ...}
    """
    if not export_audit_report:
        return {"error": "export_audit_report not available in this environment."}

    try:
        sig = inspect.signature(export_audit_report)
        params = list(sig.parameters.keys())
    except Exception:
        params = []

    # Build common kwargs we can try
    safe_kwargs = {
        "output_title": title,
        "analysis": analysis,
        "quality": quality,
        "missing": missing,
        "anomalies": anomalies,
        "drift": drift,
        "schema": schema,
        "feature_importance": feature_importance,
        "visuals": visuals,
        "generate_pdf": generate_pdf,
        "per_column_limit": per_column_limit,
    }

    # If signature expects output_path first, create a temp path
    try_orders = []

    # 1) signature accepts 'output_path' -> call with output file path + named args if allowed
    if "output_path" in params:
        # create a temporary base path
        tmp = tempfile.NamedTemporaryFile(prefix="audit_", suffix=".html", delete=False)
        tmp.close()
        output_path = tmp.name  # e.g. /tmp/audit_xxx.html
        # many versions expect output_path (without extension handling), but our report.py handles extension
        try_orders.append(("with_output_path", output_path))

    # 2) signature may expect positional (analysis, missing, quality, anomalies, drift, schema, feature_importance)
    # We'll attempt calling with positional groups if that matches param length
    try_orders.append(("positional_best", None))
    try_orders.append(("kwargs", None))

    last_exc = None
    for mode, path in try_orders:
        try:
            if mode == "with_output_path":
                # call with output_path followed by some positional or keyword args depending on params
                # try to match param order: after output_path, if next param is 'output_title' pass it
                call_args = [path]
                if "output_title" in params:
                    call_args.append(title)
                # Now try to append analysis/.. in conventional order if params include them
                # we'll use keyword args for the rest for safety
                kw = {}
                for name in ("analysis", "missing", "quality", "anomalies", "drift", "schema", "feature_importance", "visuals", "generate_pdf", "per_column_limit"):
                    if name in params:
                        kw[name] = safe_kwargs.get(name)
                res = export_audit_report(*call_args, **kw)
                return res if isinstance(res, dict) else {"result": res}
            elif mode == "positional_best":
                # attempt to call with common positional order (older versions)
                pos = [analysis, missing, quality, anomalies, drift, schema, feature_importance]
                # trim to signature length
                sig_len = len(params)
                # if function doesn't take any params, skip
                if sig_len == 0:
                    continue
                cand = pos[:sig_len]
                try:
                    res = export_audit_report(*cand)
                    return res if isinstance(res, dict) else {"result": res}
                except TypeError:
                    # try a shorter subset
                    for L in range(sig_len, 0, -1):
                        try:
                            res = export_audit_report(*pos[:L])
                            return res if isinstance(res, dict) else {"result": res}
                        except TypeError:
                            continue
                    raise
            else:
                # call by keyword args
                # filter kwargs to those supported by the signature when possible
                if params:
                    kw = {k: v for k, v in safe_kwargs.items() if k in params}
                    res = export_audit_report(**kw)
                else:
                    # unknown signature - try calling with a useful positional fallback
                    res = export_audit_report(analysis, missing, quality, anomalies, drift, schema, feature_importance)
                return res if isinstance(res, dict) else {"result": res}
        except TypeError as e:
            last_exc = e
            continue
        except Exception as e:
            return {"error": str(e)}
    return {"error": f"export_audit_report failed to match any known signature. Last error: {last_exc}"}


# ---------------------------
# Format helpers (human readable conversions)
# ---------------------------
def _missing_to_human(missing: Dict[str, Any], analysis: Dict[str, Any]) -> Tuple[List[str], pd.DataFrame]:
    out = []
    missing = _safe_dict(missing)
    per_col = _safe_dict(missing.get("per_column", {}))

    # Backwards compat: some modules return 'columns' or 'perColumn'
    if not per_col and isinstance(missing.get("columns"), dict):
        per_col = _safe_dict(missing.get("columns"))

    if not per_col:
        out.append("No missingness data available.")
        return out, pd.DataFrame(columns=["column", "missing", "missing_percent"])

    rows = []
    total_missing = 0
    for col, info in per_col.items():
        if isinstance(info, dict):
            missing_count = info.get("missing") if info.get("missing") is not None else info.get("missing_count", 0)
            missing_pct = info.get("missing_percent", info.get("missing_pct", 0))
        else:
            missing_count = int(info) if isinstance(info, (int, float)) else 0
            missing_pct = 0
        total_missing += missing_count
        rows.append({"column": col, "missing": missing_count, "missing_percent": round(float(missing_pct), 2)})

    df_missing = pd.DataFrame(rows).sort_values("missing", ascending=False)

    if df_missing["missing"].sum() == 0:
        out.append("No missing values detected.")
        return out, df_missing

    out.append(f"Total missing values across dataset: {int(df_missing['missing'].sum())}.")
    top5 = df_missing.head(5)
    out.append("Top columns by missingness:")
    for _, r in top5.iterrows():
        out.append(f"- `{r['column']}` : {int(r['missing'])} missing ({r['missing_percent']}%)")

    return out, df_missing


def _quality_to_human(quality: Dict[str, Any]) -> Tuple[List[str], pd.DataFrame]:
    out = []
    quality = _safe_dict(quality)
    if not quality:
        out.append("No quality issues detected or quality analyzer not available.")
        return out, pd.DataFrame()

    dup = quality.get("duplicate_summary", {}) or {}
    if dup:
        dup_count = dup.get("duplicate_count", dup.get("duplicate_rows", 0))
        out.append(f"Duplicate rows: {dup_count}.")
        sample = dup.get("duplicate_index_sample", []) or dup.get("duplicate_index", [])
        if sample:
            out.append(f"- Example duplicate row indices: {sample[:5]}")

    const_list = quality.get("constant_summary", {}).get("constant_columns", []) or []
    if const_list:
        out.append(f"Constant (near-constant) columns detected: {len(const_list)}.")
        for item in const_list[:5]:
            col = item.get("column", str(item))
            out.append(f"- `{col}`")

    warnings = quality.get("warnings", []) or []
    if isinstance(warnings, list) and warnings:
        out.append("Quality warnings:")
        for w in warnings[:8]:
            out.append(f"- {w}")

    rows = []
    rows.append({"metric": "duplicate_count", "value": dup.get("duplicate_count", dup.get("duplicate_rows", 0))})
    rows.append({"metric": "constant_columns_count", "value": len(const_list)})
    dfq = pd.DataFrame(rows)

    return out, dfq


def _format_feature_importance(fi: Dict[str, Any]) -> Tuple[List[str], pd.DataFrame]:
    fi = _safe_dict(fi)
    human = []
    df = pd.DataFrame()

    top_features = fi.get("top_features")
    if isinstance(top_features, list) and top_features:
        human.append("Top predictive features:")
        rows = []
        for item in top_features:
            if isinstance(item, dict):
                col = item.get("column") or item.get("name") or str(item)
                imp = item.get("importance", 0)
            else:
                col = str(item)
                imp = 0
            rows.append({"column": col, "importance": float(imp)})
            human.append(f"- `{col}` → importance {imp:.6f}")
        df = pd.DataFrame(rows).sort_values("importance", ascending=False)
        return human, df

    methods = fi.get("methods", fi.get("method", {}))
    if isinstance(methods, dict) and methods:
        human.append("Feature importance methods detected:")
        rows = []
        for method_name, detail in methods.items():
            human.append(f"- Method: `{method_name}`")
            imps = _safe_dict(detail.get("importances", {}))
            if not imps and isinstance(detail.get("importance"), dict):
                imps = _safe_dict(detail.get("importance", {}))
            # flatten fallback
            if not imps:
                for k, v in detail.items():
                    if isinstance(v, dict) and all(isinstance(x, (int, float)) for x in v.values()):
                        imps.update(v)
            for c, imp in imps.items():
                try:
                    rows.append({"method": method_name, "column": c, "importance": float(imp)})
                except Exception:
                    rows.append({"method": method_name, "column": c, "importance": 0.0})
        if rows:
            df = pd.DataFrame(rows).sort_values("importance", ascending=False)
            topn = df.groupby("column")["importance"].mean().sort_values(ascending=False).head(10)
            human.append("Aggregate top features across methods:")
            for col, val in topn.items():
                human.append(f"- `{col}` → {val:.6f}")
            return human, df

    if fi and all(isinstance(v, (int, float)) for v in fi.values()):
        rows = [{"column": k, "importance": float(v)} for k, v in fi.items()]
        df = pd.DataFrame(rows).sort_values("importance", ascending=False)
        human.append("Feature importances:")
        for _, r in df.head(10).iterrows():
            human.append(f"- `{r['column']}` → {r['importance']:.6f}")
        return human, df

    human.append("No feature importance data available.")
    return human, df


def _anomalies_to_human(anom: Dict[str, Any]) -> List[str]:
    out = []
    anom = _safe_dict(anom)
    methods = _safe_dict(anom.get("methods", {}))
    if not methods:
        out.append("No anomalies detected or anomaly module missing.")
        return out
    out.append("Anomaly detection methods ran:")
    for m in methods.keys():
        out.append(f"- {m}")
    for method_name, detail in methods.items():
        per = _safe_dict(detail.get("per_column", {}))
        if per:
            counts = {}
            for c, d in per.items():
                if isinstance(d, dict):
                    counts[c] = int(d.get("count", d.get("anomaly_count", 0)))
                elif isinstance(d, (int, float)):
                    counts[c] = int(d)
                else:
                    counts[c] = 0
            total = sum(counts.values())
            out.append(f"  - Method `{method_name}` flagged {total} anomalies across columns.")
    return out


def _drift_to_human(drift: Dict[str, Any]) -> List[str]:
    drift = _safe_dict(drift)
    if not drift:
        return ["No dataset drift detected or drift module missing."]
    out = []
    by_col = drift.get("per_column") or drift.get("drift_by_column") or drift.get("columns") or {}
    if isinstance(by_col, dict) and by_col:
        flagged = [c for c, v in by_col.items() if (isinstance(v, dict) and (v.get("drift_detected") or v.get("drift"))) or (v is True)]
        out.append(f"Columns suspected of distributional drift: {len(flagged)}")
        for c in flagged[:10]:
            out.append(f"- `{c}`")
    else:
        out.append("Drift info available.")
    return out


def _generate_actionable_recommendations(analysis, missing, anomalies, drift, quality, fi) -> List[str]:
    out = []
    missing_lines, df_missing = _missing_to_human(missing, analysis)
    if df_missing is None or df_missing.empty or df_missing["missing"].sum() == 0:
        out.append("No missing values — no imputation needed.")
    else:
        high = df_missing[df_missing["missing_percent"] > 20]
        if not high.empty:
            out.append("High-missingness columns (>20%): consider dropping or imputing:")
            for _, r in high.iterrows():
                out.append(f"- `{r['column']}` ({r['missing_percent']}% missing)")
        else:
            out.append("Missing values present but generally low per column — consider imputation strategies.")

    q_lines, qdf = _quality_to_human(quality)
    if qdf is not None and not qdf.empty:
        dup_count = int(qdf.loc[qdf["metric"] == "duplicate_count", "value"].sum()) if "duplicate_count" in qdf["metric"].values else 0
        if dup_count > 0:
            out.append(f"Found {dup_count} duplicate rows — consider deduplication.")

    anom_lines = _anomalies_to_human(anomalies)
    if any("flagged" in l or "flagged" in " ".join(anom_lines) for l in anom_lines):
        out.append("Anomalies detected — inspect top anomalous rows and consider capping/outlier handling.")

    fi_lines, fi_df = _format_feature_importance(fi)
    if fi_df is not None and not fi_df.empty:
        top = list(fi_df["column"].unique()[:5])
        out.append(f"Top predictive features to inspect for leakage/correlation: {', '.join([f'`{c}`' for c in top])}")

    drift_lines = _drift_to_human(drift)
    if any("drift" in s.lower() for s in drift_lines):
        out.append("Distribution drift detected — validate data collection and consider retraining.")

    if not out:
        out.append("No specific recommendations could be generated.")

    return out


# ---------------------------
# Main UI
# ---------------------------
st.set_page_config(layout="wide")
st.title("🔍 Explore Data — AutoDataset-Lab")

df = st.session_state["df"]
reference_df = st.session_state["reference_df"]

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
    except Exception:
        st.sidebar.error("Failed to load reference dataset.")
generate_report_flag = st.sidebar.checkbox("Enable HTML/PDF report generation", value=True)

# Run button
if st.sidebar.button("Run EDA"):
    # reset session subkeys
    for k in ["analysis", "missing", "quality", "anomalies", "drift", "schema", "feature_importance", "visuals", "recommendations", "last_report_paths"]:
        st.session_state[k] = {}
    st.session_state["eda_done"] = False

    with st.spinner("Running EDA pipeline (defensive mode)… this may take a while on large datasets"):
        sample_n = min(int(max_rows), len(df))
        df_sample = df.sample(n=sample_n, random_state=42) if sample_n < len(df) else df.copy()

        # DEBUG prints only when DEBUG True
        if DEBUG:
            try:
                if summarize_missingness:
                    print("DEBUG - MISSINGNESS:", summarize_missingness(df_sample))
                if compute_data_quality_report:
                    print("DEBUG - QUALITY:", compute_data_quality_report(df_sample))
                if detect_anomalies:
                    print("DEBUG - ANOMALIES:", detect_anomalies(df_sample))
                if detect_dataset_drift and isinstance(st.session_state.get("reference_df"), pd.DataFrame):
                    print("DEBUG - DRIFT:", detect_dataset_drift(df_sample, st.session_state.get("reference_df")))
                if compute_feature_importance_all:
                    print("DEBUG - FEATURE IMPORTANCE:", compute_feature_importance_all(df_sample, target_col if target_col else None))
            except Exception:
                pass

        # 1) Analysis
        if analyze_df:
            try:
                analysis = analyze_df(df_sample) or {}
                st.session_state["analysis"] = analysis
                # build schema defensively
                schema = {}
                cols_over = analysis.get("columns_overview", {}) or {}
                for c in df_sample.columns:
                    colinfo = cols_over.get(c, {})
                    dtype = colinfo.get("dtype") if isinstance(colinfo, dict) else str(df_sample[c].dtype)
                    inferred = None
                    if isinstance(colinfo, dict):
                        inferred = colinfo.get("semantic", {}).get("inferred_type") if isinstance(colinfo.get("semantic", {}), dict) else colinfo.get("inferred_type")
                    schema[c] = {"dtype": dtype, "inferred_type": inferred}
                st.session_state["schema"] = schema
            except Exception:
                st.session_state["analysis"] = {"rows": len(df_sample), "columns": len(df_sample.columns), "columns_overview": {}}
                st.session_state["schema"] = {c: {"dtype": str(df_sample[c].dtype), "inferred_type": None} for c in df_sample.columns}
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
            per = {}
            for c in df_sample.columns:
                miss = int(df_sample[c].isna().sum())
                pct = round(100 * miss / len(df_sample), 2) if len(df_sample) else 0
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
        if detect_dataset_drift and isinstance(st.session_state.get("reference_df"), pd.DataFrame):
            try:
                st.session_state["drift"] = detect_dataset_drift(df_sample, st.session_state["reference_df"]) or {}
            except Exception:
                st.session_state["drift"] = {}
        else:
            st.session_state["drift"] = {}

        # 6) Feature importance
        if compute_feature_importance_all:
            try:
                st.session_state["feature_importance"] = compute_feature_importance_all(df_sample, target_col if target_col else None) or {}
            except Exception:
                st.session_state["feature_importance"] = {}
        else:
            st.session_state["feature_importance"] = {}

        # 7) Visuals
        if generate_visual_bundle:
            try:
                visuals = generate_visual_bundle(df_sample, st.session_state.get("missing", {}), st.session_state.get("quality", {}), st.session_state.get("anomalies", {}), st.session_state.get("drift", {}), st.session_state.get("feature_importance", {}))
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
                    st.session_state["visuals"] = {}
            except Exception as e:
                st.session_state["visuals"] = {"__error__": f"generate_visual_bundle failed: {e}"}
        else:
            st.session_state["visuals"] = {}

        # 8) Recommendations
        st.session_state["recommendations"] = _generate_actionable_recommendations(
            st.session_state.get("analysis", {}),
            st.session_state.get("missing", {}),
            st.session_state.get("anomalies", {}),
            st.session_state.get("drift", {}),
            st.session_state.get("quality", {}),
            st.session_state.get("feature_importance", {}),
        )

        # 9) Report generation (best-effort, flexible calling)
        if generate_report_flag:
            try:
                paths = _call_export_report_flexible(
                    title=f"Audit report - {time.strftime('%Y-%m-%d %H:%M')}",
                    analysis=st.session_state.get("analysis", {}),
                    missing=st.session_state.get("missing", {}),
                    quality=st.session_state.get("quality", {}),
                    anomalies=st.session_state.get("anomalies", {}),
                    drift=st.session_state.get("drift", {}),
                    schema=st.session_state.get("schema", {}),
                    feature_importance=st.session_state.get("feature_importance", {}),
                    visuals=st.session_state.get("visuals", {}),
                    generate_pdf=False,
                )
            except Exception as e:
                paths = {"error": str(e)}
            st.session_state["last_report_paths"] = paths

    st.session_state["eda_done"] = True
    st.success("EDA complete. Scroll tabs for details.")


# ---------------------------
# Metrics row (compact)
# ---------------------------
analysis = _safe_dict(st.session_state.get("analysis", {}))
rows = analysis.get("rows", len(df))
cols = analysis.get("columns", len(df.columns))

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
        exec_lines.append(f"Dataset contains **{rows} rows** and **{cols} columns**.")
        miss_lines, miss_df = _missing_to_human(st.session_state.get("missing", {}), st.session_state.get("analysis", {}))
        if miss_df is not None and not miss_df.empty:
            exec_lines.append(f"Total missing values: **{int(miss_df['missing'].sum())}**. Top missing columns: " +
                              ", ".join([f"`{c}`" for c in (miss_df.head(3)["column"].tolist())]) + ".")
        else:
            exec_lines.append("No missing values detected.")

        q_lines, qdf = _quality_to_human(st.session_state.get("quality", {}))
        if qdf is not None and not qdf.empty:
            exec_lines.append("Data quality issues found (duplicates / constant cols).")
        else:
            exec_lines.append("No major data quality issues detected.")

        anom_lines = _anomalies_to_human(st.session_state.get("anomalies", {}))
        drift_lines = _drift_to_human(st.session_state.get("drift", {}))

        for l in exec_lines:
            st.markdown(f"- {l}")

        if anom_lines:
            st.markdown("**Anomalies:**")
            for l in anom_lines[:5]:
                st.markdown(f"- {l}")
        if drift_lines:
            st.markdown("**Drift:**")
            for l in drift_lines[:5]:
                st.markdown(f"- {l}")

# -------------
# SCHEMA TAB
# -------------
with tabs[1]:
    schema = _safe_dict(st.session_state.get("schema", {}))
    if not schema:
        st.info("No schema available.")
    else:
        df_schema = pd.DataFrame([
            {"column": col, "dtype": info.get("dtype"), "semantic": info.get("inferred_type")}
            for col, info in schema.items()
        ])
        st.dataframe(df_schema)

# -------------
# MISSING TAB
# -------------
with tabs[2]:
    st.subheader("Missingness Overview")
    miss_lines, df_missing = _missing_to_human(st.session_state.get("missing", {}), st.session_state.get("analysis", {}))
    for l in miss_lines:
        st.markdown(f"- {l}")
    if df_missing is not None and not df_missing.empty:
        st.markdown("### Missingness table (top 50)")
        st.dataframe(df_missing.head(50))

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
                rows.append({"column": c, "anomaly_count": count})
            if rows:
                st.dataframe(pd.DataFrame(rows).sort_values("anomaly_count", ascending=False))

# -------------
# DRIFT TAB
# -------------
with tabs[4]:
    st.subheader("Dataset Drift")
    drift = _safe_dict(st.session_state.get("drift", {}))
    if not drift:
        st.info("No drift data.")
    else:
        for l in _drift_to_human(drift):
            st.markdown(f"- {l}")
        with st.expander("See raw drift JSON"):
            st.json(drift)

# -------------
# QUALITY TAB
# -------------
with tabs[5]:
    st.subheader("Data Quality")
    quality = _safe_dict(st.session_state.get("quality", {}))
    q_lines, qdf = _quality_to_human(quality)
    for l in q_lines:
        st.markdown(f"- {l}")
    if qdf is not None and not qdf.empty:
        st.markdown("### Quality summary table")
        st.dataframe(qdf)

# -------------
# FEATURE IMPORTANCES TAB
# -------------
with tabs[6]:
    st.subheader("Feature Importances")
    fi = _safe_dict(st.session_state.get("feature_importance", {}))
    fi_lines, fi_df = _format_feature_importance(fi)
    for l in fi_lines:
        st.markdown(f"- {l}")
    if fi_df is not None and not fi_df.empty:
        st.markdown("### Feature importance table (top 200)")
        st.dataframe(fi_df.head(200))

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
        st.info("No visuals produced. Ensure `core.eda.visualize.generate_visual_bundle` exists and returns fig objects or a mapping.")
    else:
        if isinstance(visuals, dict):
            items = visuals.items()
        elif isinstance(visuals, list):
            items = []
            for i, item in enumerate(visuals):
                if isinstance(item, tuple) and len(item) == 2:
                    items.append((str(item[0]), item[1]))
                else:
                    items.append((f"visual_{i}", item))
        else:
            items = [("__visual__", visuals)]

        any_rendered = False
        for title, fig in items:
            if title == "__error__":
                st.warning(fig)
                continue
            st.markdown(f"### {title}")
            rendered = False
            # matplotlib figure
            try:
                import matplotlib.pyplot as plt
                from matplotlib.figure import Figure
                if isinstance(fig, Figure):
                    st.pyplot(fig)
                    rendered = True
            except Exception:
                pass
            if rendered:
                any_rendered = True
                continue
            # plotly figure
            try:
                import plotly.graph_objs as go  # noqa: F401
                # simple duck type
                if hasattr(fig, "to_plotly_json") or "plotly" in str(type(fig)).lower():
                    st.plotly_chart(fig, use_container_width=True)
                    rendered = True
            except Exception:
                pass
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
                from PIL import Image
                if isinstance(fig, Image.Image):
                    st.image(fig)
                    any_rendered = True
                    continue
            except Exception:
                pass
            # fallback
            st.write("Cannot render visual of type:", type(fig))
            st.write(fig)

        if not any_rendered:
            st.info("No visual objects were renderable. Check the return type of generate_visual_bundle.")

# End of file

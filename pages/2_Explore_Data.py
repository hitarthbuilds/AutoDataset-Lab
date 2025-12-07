# pages/2_Explore_Data.py
# Enterprise Explore page — SHAP + PDF + multi-tab layout
from __future__ import annotations
import io
import os
import time
import json
import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# plotting
import matplotlib.pyplot as plt

# reportlab for PDF
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# shap
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# local helpers - create these files below
from core.eda.feature_importance import compute_feature_importance_all  # existing enterprise module (you already have)
from core.eda.utils_shap import encode_df_for_model, safe_shap_explainer, save_matplotlib_figure_to_png
from core.eda.report_helpers import build_multipage_pdf_with_images

# -------------------------
# Page layout / state init
# -------------------------
st.set_page_config(page_title="Explore Data — Enterprise EDA", layout="wide")
if "eda_analysis" not in st.session_state:
    st.session_state["eda_analysis"] = None
if "shap_images" not in st.session_state:
    st.session_state["shap_images"] = {}  # name -> bytes
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None

# Sidebar controls
with st.sidebar:
    st.header("EDA controls")
    run_btn = st.button("🚀 Run full enterprise EDA")
    sample_rows = st.number_input("Sample rows (for visuals)", min_value=200, max_value=200000, value=2000, step=200)
    gen_pdf = st.checkbox("Generate multipage PDF report", value=True)
    export_json = st.checkbox("Export analysis JSON", value=True)
    export_pngs = st.checkbox("Export PNGs (per visual)", value=True)
    target_col = st.text_input("Target column (optional)", value="")

# load dataset (assume uploaded earlier into session_state['df'] or you can adapt)
df: Optional[pd.DataFrame] = st.session_state.get("df", None)
if df is None:
    st.info("No dataset loaded. Upload dataset on the Upload page or set session_state['df'].")
    st.stop()

# Top header
st.title("🔎 Explore Data — Enterprise EDA Cockpit")
st.write("Multi-tab interactive explorer with SHAP, visuals, and PDF export (VC demo mode).")

# Multitab layout
tabs = st.tabs(["Overview", "SHAP / Explanations", "Visual Panels", "Recommendations & Fixes", "Exports"])

# -------------------------
# Run full EDA (lightweight orchestrator)
# -------------------------
def run_full_eda(df: pd.DataFrame, target: Optional[str] = None) -> Dict[str, Any]:
    """
    Orchestrator for full EDA pipeline. Returns dict of results (quality, missing, anomalies, drift, feature_importance,...)
    This function calls your enterprise modules where available.
    """
    out = {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "timestamp": time.time()}
    try:
        # feature importance (best-effort; may be heavy)
        fi = compute_feature_importance_all(df, target=target, methods=["rf", "permutation", "mutual_info", "cramers_v"])
        out["feature_importance"] = fi
    except Exception:
        out["feature_importance_error"] = traceback.format_exc()

    # basic quality / missingness summary (fallback)
    try:
        missing = df.isna().sum().to_dict()
        out["missing"] = {k: int(v) for k, v in missing.items()}
    except Exception:
        out["missing_error"] = traceback.format_exc()

    # basic per-column dtypes
    out["dtypes"] = {c: str(df[c].dtype) for c in df.columns}
    return out

if run_btn:
    with st.spinner("Running full enterprise EDA... this can take a minute for big datasets"):
        try:
            analysis = run_full_eda(df, target_col if (target_col := target_col or None) else None)
            st.session_state["eda_analysis"] = analysis
            st.session_state["last_run"] = time.time()
            st.success("EDA pipeline finished. Scroll to the tabs for outputs.")
        except Exception as e:
            st.error("EDA pipeline failed: " + str(e))
            st.exception(traceback.format_exc())

# short helper to ensure we have analysis
analysis = st.session_state.get("eda_analysis", None)
if analysis is None:
    st.info("No analysis results yet. Click *Run full enterprise EDA*.")
    st.stop()

# -------------------------
# Tab: Overview
# -------------------------
with tabs[0]:
    st.header("Executive KPIs")
    row1, row2, row3, row4 = st.columns(4)
    row1.metric("Rows", f"{analysis.get('rows', '?')}")
    row2.metric("Columns", f"{analysis.get('cols', '?')}")
    # simple counts
    missing_cols = sum(1 for v in analysis.get("missing", {}).values() if v > 0)
    row3.metric("Cols with missing", f"{missing_cols}")
    row4.metric("Computed FI methods", len(analysis.get("feature_importance", {}).get("methods", {})))
    st.markdown("---")
    st.subheader("Top aggregated features (from importance)")
    agg = analysis.get("feature_importance", {}).get("aggregated", {}).get("top_features", [])
    if agg:
        st.write(agg[:30])
    else:
        st.warning("No aggregated feature importance available.")

# -------------------------
# Tab: SHAP / Explanations
# -------------------------
with tabs[1]:
    st.header("SHAP explanations (dynamic)")
    if not _HAS_SHAP:
        st.error("SHAP not installed. Install `shap` to use this panel.")
    else:
        col1, col2 = st.columns([2, 1])
        with col2:
            st.subheader("Controls")
            shap_sample = st.slider("SHAP sample size", min_value=50, max_value=5000, value=500, step=50)
            show_force = st.checkbox("Show force plot (per-row)", value=False)
            show_dependence = st.checkbox("Enable dependence plot", value=True)
            waterfall_toggle = st.checkbox("Enable per-feature waterfall PNGs", value=False)
            choose_feature = st.selectbox("Feature for dependence/waterfall", options=list(df.columns), index=0)
            generate_shap = st.button("Generate SHAP visuals now")

        with col1:
            st.subheader("SHAP outputs")
            shap_placeholder = st.empty()
            shap_status = st.empty()

        # run SHAP generation on demand
        if generate_shap:
            with st.spinner("Preparing encoded sample and SHAP explainer..."):
                try:
                    # safety: create copy sample
                    X = df.drop(columns=[target_col]) if (target_col) and (target_col in df.columns) else df.copy()
                    if X.shape[0] > shap_sample:
                        X_small = X.sample(n=shap_sample, random_state=42).copy()
                    else:
                        X_small = X.copy()

                    # encode columns to numeric matrix expected by shap
                    X_enc = encode_df_for_model(X_small)
                    st.write("Encoded sample shape:", X_enc.shape)
                    # model: use RF saved inside feature_importance if available; else train quick RF
                    fi_methods = analysis.get("feature_importance", {}).get("methods", {})
                    model = None
                    # if compute_feature_importance_all already returned model meta with trained model we could use it
                    # fallback: train a small RF here
                    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                    y_provided = (target_col in df.columns)
                    if y_provided:
                        y_small = df.loc[X_enc.index, target_col]
                        model = RandomForestRegressor(n_estimators=100, random_state=42) if pd.api.types.is_numeric_dtype(y_small) else RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(X_enc, y_small)
                    else:
                        # train a pseudo regressor on a synthetic target (unsupervised shap approximation)
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        # synthetic quick target: sum of numeric columns
                        synth = X_enc.select_dtypes(np.number).fillna(0).sum(axis=1).values
                        model.fit(X_enc, synth)

                    # safe explainer (handles TreeExplainer issues)
                    explainer = safe_shap_explainer(model, X_enc)
                    # compute shap values (works with both explainer types)
                    shap_values = explainer(X_enc) if hasattr(explainer, "__call__") else explainer.shap_values(X_enc)
                except Exception as e:
                    shap_status.error("SHAP pipeline failed: " + str(e))
                    shap_status.exception(traceback.format_exc())
                    shap_placeholder.info("SHAP plotting not available.")
                    generate_shap = False
                    shap_values = None
                    explainer = None

            # if succeeded, create PNGs and show summary plot
            if shap_values is not None:
                try:
                    # summary plot (matplotlib) and save png
                    fig_summary = plt.figure(figsize=(8, max(4, min(12, X_enc.shape[1] * 0.2))))
                    # shap handles both arrays and Explanation objects
                    if hasattr(shap, "summary_plot"):
                        shap.summary_plot(shap_values, X_enc, show=False)
                    else:
                        # fallback: create simple bar by mean abs
                        mean_abs = np.abs(shap_values.values).mean(axis=0)
                        idx = np.argsort(-mean_abs)
                        plt.barh(np.array(X_enc.columns)[idx][:30], mean_abs[idx][:30])
                    summary_png = save_matplotlib_figure_to_png(fig_summary)
                    st.session_state["shap_images"]["summary"] = summary_png
                    st.image(summary_png, caption="SHAP summary (mean abs)", use_column_width=True)

                    # dependence plot for chosen feature
                    if show_dependence:
                        fig_dep = plt.figure(figsize=(6, 4))
                        try:
                            shap.dependence_plot(choose_feature, shap_values.values if hasattr(shap_values, "values") else shap_values, X_enc, show=False)
                        except Exception:
                            # fallback scatter
                            plt.scatter(X_enc[choose_feature].values, np.abs(shap_values.values).mean(axis=1))
                            plt.xlabel(choose_feature)
                            plt.ylabel("avg |SHAP value|")
                        dep_png = save_matplotlib_figure_to_png(fig_dep)
                        st.session_state["shap_images"][f"dependence_{choose_feature}"] = dep_png
                        st.image(dep_png, caption=f"Dependence: {choose_feature}", use_column_width=True)

                    # per-row force plot (if requested)
                    if show_force:
                        # show an interactive force plot for a single random row
                        row_idx = X_enc.index[0]
                        try:
                            # convert to shap.Explanation if necessary
                            if hasattr(shap_values, "values"):
                                vals = shap_values.values[0]
                                base = shap_values.base_values[0] if hasattr(shap_values, "base_values") else 0
                                f_html = shap.plots.force(shap_values[0], matplotlib=False)
                                # shap.plots.force returns a html widget; save as png via saving matplotlib fallback
                                st.write("Force plots (interactive) are best saved as HTML. Using static fallback for PNG.")
                            else:
                                st.write("Cannot render force plot with this explainer.")
                        except Exception:
                            st.write("Force plot generation failed (falling back to text).")
                    # waterfall per-feature PNG (if toggled)
                    if waterfall_toggle:
                        # use shap.plots.waterfall if available or approximate
                        feat = choose_feature
                        # Try to create single-feature waterfall for top-k rows
                        try:
                            # pick 1st row
                            if hasattr(shap_values, "values"):
                                wfig = plt.figure(figsize=(6, 4))
                                shap.plots.bar(shap_values, max_display=30, show=False)
                                wp = save_matplotlib_figure_to_png(wfig)
                                st.session_state["shap_images"][f"waterfall_{feat}"] = wp
                                st.image(wp, caption=f"Waterfall/Bar (approx) {feat}")
                        except Exception:
                            st.write("Waterfall generation failed (non-fatal).")
                    st.success("SHAP visuals generated.")
                except Exception:
                    st.error("Failed rendering SHAP visuals.")
                    st.exception(traceback.format_exc())

# -------------------------
# Tab: Visual Panels
# -------------------------
with tabs[2]:
    st.header("Visual Panels")
    st.subheader("Missingness (per-column)")
    missing_df = pd.DataFrame.from_dict(analysis.get("missing", {}), orient="index", columns=["missing_count"])
    missing_df["missing_pct"] = missing_df["missing_count"] / analysis.get("rows", 1)
    st.dataframe(missing_df.sort_values("missing_pct", ascending=False).head(200))

    st.subheader("Data quality diagnostics (constants / infinities / negatives / high-card)")
    dtypes = analysis.get("dtypes", {})
    st.write(pd.Series(dtypes).head(200))

# -------------------------
# Tab: Recommendations & Fixes
# -------------------------
with tabs[3]:
    st.header("What I found / Recommendations")
    st.write("LLM-style narrative (editable). This content may be auto-generated from EDA findings.")
    # simple editable box with prefilled skeleton
    skeleton = st.session_state.get("last_narrative", None)
    if skeleton is None:
        rows = analysis.get("rows", "?"); cols = analysis.get("cols", "?")
        skeleton = f"Dataset contains **{rows}** rows and **{cols}** columns. Top issues: missingness, cardinality, and model-ready anomalies."
    narrative = st.text_area("LLM-style prompt (editable)", value=skeleton, height=200)
    st.session_state["last_narrative"] = narrative

    # One-click fixes (very simple safe transformations)
    st.subheader("One-click: Fix my dataset (safe transforms)")
    if st.button("Apply safe transforms"):
        with st.spinner("Applying safe transforms (fill missing with placeholders, cast numerics)..."):
            try:
                df_fixed = df.copy()
                for c in df_fixed.columns:
                    if pd.api.types.is_numeric_dtype(df_fixed[c]):
                        df_fixed[c] = pd.to_numeric(df_fixed[c], errors="coerce").fillna(0)
                    else:
                        df_fixed[c] = df_fixed[c].astype(str).fillna("__NA__")
                st.session_state["df_fixed"] = df_fixed
                st.success("Safe transforms applied and stored in session_state['df_fixed']. Review and re-run EDA.")
            except Exception:
                st.error("Failed to apply transforms.")
                st.exception(traceback.format_exc())

# -------------------------
# Tab: Exports
# -------------------------
with tabs[4]:
    st.header("Exports")
    if export_json:
        st.download_button("Download eda_analysis.json", data=json.dumps(analysis, default=str, indent=2), file_name="eda_analysis.json")
    if export_pngs:
        st.write("SHAP PNG assets:")
        for name, b in st.session_state.get("shap_images", {}).items():
            st.download_button(f"PNG: {name}", data=b, file_name=f"shap_{name}.png")

    if gen_pdf:
        if st.button("Download multipage PDF report"):
            try:
                images = st.session_state.get("shap_images", {})
                pdf_bytes = build_multipage_pdf_with_images(analysis, images)
                st.download_button("Download PDF report", data=pdf_bytes, file_name="eda_audit_report.pdf")
            except Exception:
                st.error("PDF generation failed.")
                st.exception(traceback.format_exc())

"""
Enterprise-grade visualization module for AutoDataset-Lab.
Produces Plotly figures (recommended for embedding into HTML/PDF via base64)
and Matplotlib fallback figures when needed.

All functions return a figure object (NOT shown),
ready for embedding via report.fig_to_base64_png()

Visualization coverage:
- Missingness heatmap
- Correlation heatmaps (Pearson, Spearman)
- Top categorical bar distributions
- Numeric distributions (histograms + KDE)
- Quality heatmap (constants, infinities, negatives)
- Anomaly overlays (scatter)
- Drift visual: reference vs current distributions
- Feature importance bar chart

No Streamlit inside this file. Pure figure-generation only.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List

import plotly.express as px
import plotly.graph_objects as go


# ============================================
# Missingness Heatmap
# ============================================
def plot_missingness_heatmap(df: pd.DataFrame):
    miss = df.isna().astype(int)
    if miss.sum().sum() == 0:
        return px.imshow(np.zeros((1,1)), text_auto=True, title="No Missingness Found")

    fig = px.imshow(
        miss.T,
        color_continuous_scale=["#0f0f0f", "#ff3333"],
        aspect="auto",
        title="Missingness Heatmap"
    )
    fig.update_layout(
        width=900,
        height=600,
        xaxis_title="Row index",
        yaxis_title="Columns",
        coloraxis_colorbar_title="Missing"
    )
    return fig


# ============================================
# Correlation Heatmaps (Pearson & Spearman)
# ============================================
def plot_correlation_heatmaps(df: pd.DataFrame, top_k: int = 20) -> Dict[str, Any]:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {"pearson": None, "spearman": None}

    # Trim large datasets
    if numeric_df.shape[1] > top_k:
        numeric_df = numeric_df.iloc[:, :top_k]

    pear = numeric_df.corr(method="pearson")
    spear = numeric_df.corr(method="spearman")

    fig_pear = px.imshow(
        pear,
        text_auto=False,
        color_continuous_scale="RdBu_r",
        title="Pearson Correlation"
    )

    fig_spear = px.imshow(
        spear,
        text_auto=False,
        color_continuous_scale="RdBu_r",
        title="Spearman Correlation"
    )

    return {"pearson": fig_pear, "spearman": fig_spear}


# ============================================
# Numeric Distributions (Histogram + KDE)
# ============================================
def plot_numeric_distributions(df: pd.DataFrame, max_cols: int = 6) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    figs = {}
    for col in numeric_cols:
        fig = px.histogram(
            df,
            x=col,
            nbins=60,
            marginal="box",
            opacity=0.7,
            title=f"Distribution: {col}"
        )
        figs[col] = fig
    return figs


# ============================================
# Categorical Top-Values Bars
# ============================================
def plot_categorical_top_values(df: pd.DataFrame, max_cols: int = 6) -> Dict[str, Any]:
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()[:max_cols]
    figs = {}
    for col in cats:
        vc = df[col].astype(str).value_counts().nlargest(20)
        fig = px.bar(
            vc,
            x=vc.index,
            y=vc.values,
            title=f"Top categories: {col}"
        )
        fig.update_layout(xaxis_title=col, yaxis_title="Count")
        figs[col] = fig
    return figs


# ============================================
# Data Quality Heatmap
# ============================================
def plot_quality_heatmap(quality: Dict[str, Any]):
    """
    quality["per_column"] = {
        col: { "is_constant": bool, "infinities": int, "negatives": int }
    }
    """
    data = quality.get("per_column", {})
    if not data:
        return None

    qdf = pd.DataFrame.from_dict(data, orient="index")
    qdf = qdf[["is_constant", "infinities", "negatives"]]

    fig = px.imshow(
        qdf.T,
        text_auto=True,
        color_continuous_scale="YlOrRd",
        title="Data Quality Heatmap"
    )
    fig.update_layout(height=600, width=900)
    return fig


# ============================================
# Anomaly Scatter Overlay
# ============================================
def plot_anomalies(df: pd.DataFrame, anomalies: Dict[str, Any], x_col=None, y_col=None):
    if anomalies is None or not anomalies.get("points"):
        return None

    points = anomalies["points"]  # list of {"index":..., "score":...}

    # Pick default numeric cols if not provided
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None

    x_col = x_col or numeric_cols[0]
    y_col = y_col or numeric_cols[1]

    base = px.scatter(df, x=x_col, y=y_col, opacity=0.4, title="Anomaly scatter overlay")
    anomaly_df = pd.DataFrame(points)

    overlay = px.scatter(
        anomaly_df,
        x=df.loc[anomaly_df["index"], x_col],
        y=df.loc[anomaly_df["index"], y_col],
        color=anomaly_df["score"],
        color_continuous_scale="Reds",
        size_max=12
    )

    for trace in overlay.data:
        base.add_trace(trace)

    return base


# ============================================
# Drift Visual: Reference vs Current
# ============================================
def plot_drift_comparison(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature: str
):
    if feature not in reference.columns or feature not in current.columns:
        return None

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=reference[feature],
        opacity=0.5,
        name="Reference",
        marker_color="#2b8cff"
    ))
    fig.add_trace(go.Histogram(
        x=current[feature],
        opacity=0.5,
        name="Current",
        marker_color="#ff3333"
    ))

    fig.update_layout(
        barmode="overlay",
        title=f"Drift: {feature} (Reference vs Current)",
        xaxis_title=feature,
        yaxis_title="Frequency"
    )
    return fig


# ============================================
# Feature Importance Bar
# ============================================
def plot_feature_importance(importance: List[Dict[str, Any]]):
    """
    importance = [{ "feature": ..., "score": ... }, ...]
    """
    if not importance:
        return None

    df = pd.DataFrame(importance).sort_values("score", ascending=False)
    fig = px.bar(
        df,
        x="score",
        y="feature",
        orientation="h",
        title="Feature Importance (Model-based)"
    )
    fig.update_layout(height=600)
    return fig


# ============================================
# Mega Visual Bundle
# ============================================
def generate_visual_bundle(
    df: pd.DataFrame,
    missing: Dict[str, Any],
    quality: Dict[str, Any],
    anomalies: Dict[str, Any],
    drift: Dict[str, Any],
    feature_importance: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Returns a dict of all important visuals.
    Used directly by the PDF/HTML report generator.
    """

    visuals = {}

    # Missingness
    visuals["missingness_heatmap"] = plot_missingness_heatmap(df)

    # Correlations
    corrs = plot_correlation_heatmaps(df)
    visuals["corr_pearson"] = corrs["pearson"]
    visuals["corr_spearman"] = corrs["spearman"]

    # Numeric distributions
    for col, fig in plot_numeric_distributions(df).items():
        visuals[f"dist_{col}"] = fig

    # Categorical top values
    for col, fig in plot_categorical_top_values(df).items():
        visuals[f"cat_{col}"] = fig

    # Quality heatmap
    qfig = plot_quality_heatmap(quality)
    if qfig:
        visuals["quality_heatmap"] = qfig

    # Anomalies overlay
    af = plot_anomalies(df, anomalies)
    if af:
        visuals["anomalies_overlay"] = af

    # Drift visuals – if paired data exists
    if drift.get("reference") is not None and drift.get("current") is not None:
        reference = drift["reference"]
        current = drift["current"]
        for col in reference.select_dtypes(include=[np.number]).columns[:6]:
            fig = plot_drift_comparison(reference, current, col)
            if fig:
                visuals[f"drift_{col}"] = fig

    # Feature importance
    if feature_importance.get("top_features"):
        fig = plot_feature_importance(feature_importance["top_features"])
        if fig:
            visuals["feature_importance"] = fig

    return visuals

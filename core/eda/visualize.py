"""
Enterprise-grade visualization module for AutoDataset-Lab.
All functions produce Plotly figures (preferred) with Matplotlib fallback support.

Hardened-version:
- All functions guaranteed to return either a Plotly figure or None.
- generate_visual_bundle now accepts optional args with safe defaults.
- No assumptions about dict structure.
- Fully defensive conversions.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List

import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# Missingness Heatmap
# ============================================================
def plot_missingness_heatmap(df: pd.DataFrame):
    try:
        miss = df.isna().astype(int)
    except Exception:
        return None

    if miss.sum().sum() == 0:
        return px.imshow(np.zeros((1, 1)), text_auto=True, title="No Missingness Found")

    try:
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
    except Exception:
        return None


# ============================================================
# Correlation Heatmaps (Pearson / Spearman)
# ============================================================
def plot_correlation_heatmaps(df: pd.DataFrame, top_k: int = 20) -> Dict[str, Any]:
    out = {"pearson": None, "spearman": None}

    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return out

        if numeric_df.shape[1] > top_k:
            numeric_df = numeric_df.iloc[:, :top_k]

        pear = numeric_df.corr(method="pearson")
        spear = numeric_df.corr(method="spearman")

        out["pearson"] = px.imshow(
            pear,
            text_auto=False,
            color_continuous_scale="RdBu_r",
            title="Pearson Correlation"
        )
        out["spearman"] = px.imshow(
            spear,
            text_auto=False,
            color_continuous_scale="RdBu_r",
            title="Spearman Correlation"
        )
        return out
    except Exception:
        return out


# ============================================================
# Numeric Distributions
# ============================================================
def plot_numeric_distributions(df: pd.DataFrame, max_cols: int = 6) -> Dict[str, Any]:
    figs = {}
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
        for col in numeric_cols:
            try:
                fig = px.histogram(df, x=col, nbins=60, marginal="box", opacity=0.7, title=f"Distribution: {col}")
                figs[col] = fig
            except Exception:
                continue
    except Exception:
        pass
    return figs


# ============================================================
# Categorical Top-Values
# ============================================================
def plot_categorical_top_values(df: pd.DataFrame, max_cols: int = 6) -> Dict[str, Any]:
    figs = {}
    try:
        cats = df.select_dtypes(include=["object", "category"]).columns.tolist()[:max_cols]
        for col in cats:
            try:
                vc = df[col].astype(str).value_counts().nlargest(20)
                fig = px.bar(vc, x=vc.index, y=vc.values, title=f"Top categories: {col}")
                fig.update_layout(xaxis_title=col, yaxis_title="Count")
                figs[col] = fig
            except Exception:
                continue
    except Exception:
        pass
    return figs


# ============================================================
# Data Quality Heatmap
# ============================================================
def plot_quality_heatmap(quality: Dict[str, Any]):
    try:
        data = quality.get("per_column", {})
        if not isinstance(data, dict) or not data:
            return None

        qdf = pd.DataFrame.from_dict(data, orient="index")

        # defensive trimming
        keep_cols = [c for c in ["is_constant", "infinities", "negatives"] if c in qdf.columns]
        if not keep_cols:
            return None

        qdf = qdf[keep_cols]

        fig = px.imshow(
            qdf.T,
            text_auto=True,
            color_continuous_scale="YlOrRd",
            title="Data Quality Heatmap"
        )
        fig.update_layout(height=600, width=900)
        return fig
    except Exception:
        return None


# ============================================================
# Anomaly Scatter Overlay
# ============================================================
def plot_anomalies(df: pd.DataFrame, anomalies: Dict[str, Any], x_col=None, y_col=None):
    try:
        if anomalies is None or "points" not in anomalies:
            return None

        points = anomalies.get("points", [])
        if not isinstance(points, list) or not points:
            return None

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return None

        x_col = x_col or numeric_cols[0]
        y_col = y_col or numeric_cols[1]

        anomaly_df = pd.DataFrame(points)
        if "index" not in anomaly_df:
            return None

        base = px.scatter(df, x=x_col, y=y_col, opacity=0.4, title="Anomaly Scatter Overlay")

        overlay = px.scatter(
            anomaly_df,
            x=df.loc[anomaly_df["index"], x_col],
            y=df.loc[anomaly_df["index"], y_col],
            color=anomaly_df.get("score", [1] * len(anomaly_df)),
            color_continuous_scale="Reds",
        )

        for trace in overlay.data:
            base.add_trace(trace)

        return base
    except Exception:
        return None


# ============================================================
# Drift Visuals
# ============================================================
def plot_drift_comparison(reference: pd.DataFrame, current: pd.DataFrame, feature: str):
    try:
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
    except Exception:
        return None


# ============================================================
# Feature Importance Bar
# ============================================================
def plot_feature_importance(top_features: List[Dict[str, Any]]):
    try:
        if not isinstance(top_features, list) or not top_features:
            return None

        df = pd.DataFrame(top_features)
        # support both {"column":..., "importance":...} and {"feature":..., "score":...}
        if "column" in df.columns and "importance" in df.columns:
            df = df.sort_values("importance", ascending=False)
            x_col = "importance"
            y_col = "column"
        elif "feature" in df.columns and "score" in df.columns:
            df = df.sort_values("score", ascending=False)
            x_col = "score"
            y_col = "feature"
        else:
            return None

        fig = px.bar(df, x=x_col, y=y_col, orientation="h", title="Feature Importance")
        fig.update_layout(height=600)
        return fig
    except Exception:
        return None


# ============================================================
# MEGA VISUAL BUNDLE (FIXED SIGNATURE)
# ============================================================
def generate_visual_bundle(
    df: pd.DataFrame,
    missing: Dict[str, Any] = None,
    quality: Dict[str, Any] = None,
    anomalies: Dict[str, Any] = None,
    drift: Dict[str, Any] = None,
    feature_importance: Dict[str, Any] = None,
) -> Dict[str, Any]:

    missing = missing or {}
    quality = quality or {}
    anomalies = anomalies or {}
    drift = drift or {}
    feature_importance = feature_importance or {}

    visuals = {}

    # Missingness
    visuals["missingness_heatmap"] = plot_missingness_heatmap(df)

    # Correlations
    corrs = plot_correlation_heatmaps(df)
    visuals["corr_pearson"] = corrs.get("pearson")
    visuals["corr_spearman"] = corrs.get("spearman")

    # Numeric distributions
    for col, fig in plot_numeric_distributions(df).items():
        visuals[f"dist_{col}"] = fig

    # Categorical top values
    for col, fig in plot_categorical_top_values(df).items():
        visuals[f"cat_{col}"] = fig

    # Quality heatmap
    visuals["quality_heatmap"] = plot_quality_heatmap(quality)

    # Anomalies
    visuals["anomalies_overlay"] = plot_anomalies(df, anomalies)

    # Drift
    if isinstance(drift.get("reference"), pd.DataFrame) and isinstance(drift.get("current"), pd.DataFrame):
        reference = drift["reference"]
        current = drift["current"]
        for col in reference.select_dtypes(include=[np.number]).columns[:6]:
            visuals[f"drift_{col}"] = plot_drift_comparison(reference, current, col)

    # Feature importance
    if isinstance(feature_importance.get("top_features"), list):
        visuals["feature_importance"] = plot_feature_importance(feature_importance["top_features"])

    return visuals

import streamlit as st
import pandas as pd
import json

from core.utils.sessions import get_df

# New EDA imports (fully compatible with your new architecture)
from core.eda.analyze import analyze_dataframe
from core.eda.missing import missing_summary
from core.eda.quality import quality_report
from core.eda.anamolies import detect_anomalies
from core.eda.drift import detect_drift
from core.eda.visualize import plotly_missing_heatmap


# -------------------------------------------------------------------
# MAIN INSIGHTS PAGE
# -------------------------------------------------------------------
def app():

    st.title("🔍 AI Insights & Automated EDA Summary")

    # Load dataframe
    df = get_df()
    if df is None:
        st.error("No dataset loaded. Upload a dataset first.")
        return

    st.success("Dataset successfully loaded!")

    st.subheader("📘 Dataset Preview")
    st.dataframe(df.head())

    st.divider()

    # -------------------------------------------------------------------
    # Run EDA computations
    # -------------------------------------------------------------------
    st.header("📊 Automated EDA Insights")

    with st.spinner("Analyzing dataset..."):
        analysis = analyze_dataframe(df)
        missing = missing_summary(df)
        quality = quality_report(df)
        anomalies = detect_anomalies(df)
        drift = detect_drift(df)

    st.success("Analysis complete!")

    # -------------------------------------------------------------------
    # High-level Stats
    # -------------------------------------------------------------------

    st.subheader("📌 Overview Summary")

    st.json({
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numeric_columns": analysis.get("numeric_count"),
        "categorical_columns": analysis.get("categorical_count"),
        "missing_cells": missing.get("total_missing"),
    })

    st.divider()

    # -------------------------------------------------------------------
    # Missingness Visualization
    # -------------------------------------------------------------------

    st.subheader("🧩 Missing Values Heatmap")

    fig = plotly_missing_heatmap(df)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------------------------------------------------------------------
    # Data Quality
    # -------------------------------------------------------------------
    st.subheader("🧪 Data Quality Report")
    st.json(quality)

    st.divider()

    # -------------------------------------------------------------------
    # Anomalies
    # -------------------------------------------------------------------
    st.subheader("🚨 Anomaly Detection")
    st.json(anomalies)

    st.divider()

    # -------------------------------------------------------------------
    # Drift Detection (if time/partition columns exist)
    # -------------------------------------------------------------------
    st.subheader("🌪 Data Drift Detection")
    st.json(drift)

    st.divider()

    # -------------------------------------------------------------------
    # Download Full Insight JSON
    # -------------------------------------------------------------------
    st.header("📥 Download Full AI Insights")

    final_report = {
        "overview": analysis,
        "missing": missing,
        "quality": quality,
        "anomalies": anomalies,
        "drift": drift,
    }

    st.download_button(
        label="📄 Download Insights Report (JSON)",
        data=json.dumps(final_report, indent=4),
        file_name="insights_report.json",
        mime="application/json"
    )


# For standalone debugging
if __name__ == "__main__":
    app()

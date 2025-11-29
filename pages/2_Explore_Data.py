import streamlit as st
import pandas as pd
import polars as pl
import traceback

from core.utils.sessions import get_df
from core.eda.analyze import analyze_dataframe
from core.eda.visualize import (
    plot_numeric_distribution,
    plot_categorical_distribution,
    plot_correlation_heatmap
)
from core.eda.report import generate_visual_eda_report


def app():
    st.header("🔍 Explore Data")

    # Load dataset
    df_polars = get_df()
    if df_polars is None:
        st.warning("Upload a dataset first.")
        return

    df = df_polars.to_pandas()

    try:
        # -------------------------------
        # Preview
        #-------------------------------
        st.subheader("📄 Preview")
        st.dataframe(df.head(), use_container_width=True)

        # -------------------------------
        # Analysis summary
       # -------------------------------
        analysis = analyze_dataframe(df)

        with st.expander("📌 Overview"):
            st.json(analysis["overview"])

        with st.expander("⚠ Missing Values"):
            st.json(analysis["missing"])

        with st.expander("📊 Numeric Summary"):
            st.json(analysis["numeric_summary"])

        with st.expander("🧩 Categorical Summary"):
            st.json(analysis["categorical_summary"])

        # -------------------------------
        # Visualizations
        # -------------------------------
        st.subheader("📈 Visualizations")

        numerics = analysis["column_types"]["numeric"]
        cats = analysis["column_types"]["categorical"]

        col1, col2 = st.columns(2)

        # NUMERIC VISUALS
        if numerics:
            st.markdown("### 🔢 Numeric Distributions")
            for i, col in enumerate(numerics):
                target = col1 if i % 2 == 0 else col2
                with target:
                    st.markdown(f"**{col}**")
                    fig = plot_numeric_distribution(df, col)
                    fig.set_size_inches(4, 3)  # compact
                    st.pyplot(fig)

        # CATEGORICAL VISUALS
        if cats:
            st.markdown("### 🔤 Categorical Distributions")
            for i, col in enumerate(cats):
                target = col1 if i % 2 == 0 else col2
                with target:
                    st.markdown(f"**{col}**")
                    fig = plot_categorical_distribution(df, col)
                    fig.set_size_inches(4, 3)
                    st.pyplot(fig)

        # CORRELATION MATRIX
        if len(numerics) > 1:
            st.markdown("### 🔥 Correlation Heatmap")
            fig = plot_correlation_heatmap(df[numerics])
            fig.set_size_inches(6, 4)
            st.pyplot(fig)

        # -------------------------------
        # DOWNLOAD REPORT
       # -------------------------------
        st.markdown("---")
        st.subheader("📥 Export EDA Report")

        pdf_path = generate_visual_eda_report(df)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download Report",
                data=f,
                file_name="eda_report.pdf",
                mime="application/pdf"
            )

        st.success("EDA PDF report generated successfully.")

    except Exception as e:
        st.error("Something went wrong inside Explore Data.")
        st.code(traceback.format_exc())


# IMPORTANT
# This MUST be here or Streamlit won't render the page
app()

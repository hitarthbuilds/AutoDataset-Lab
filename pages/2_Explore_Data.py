import streamlit as st
import polars as pl

from core.utils.session import get_df
from core.eda.summary import dataset_overview
from core.eda.missing_value import missing_summary
from core.eda.correlations import compute_correlations

st.title("Explore Data")

df = get_df()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Missing Values", "Distributions", "Correlations"])

with tab1:
    st.subheader("Dataset Overview")
    st.write(dataset_overview(df))

    st.subheader("Preview (First 100 Rows)")
    st.dataframe(df.head(100))

with tab2:
    st.subheader("Missing Value Summary")
    st.dataframe(missing_summary(df))

with tab3:
    st.subheader("Distributions")
    numeric_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt in (pl.Int64, pl.Float64)]
    if numeric_cols:
        for col in numeric_cols:
            st.bar_chart(df[col])
    else:
        st.info("No numeric columns found.")

with tab4:
    st.subheader("Correlations")
    corr = compute_correlations(df)
    st.dataframe(corr)

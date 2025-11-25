import streamlit as st
import pandas as pd
import polars as pl

from core.utils.file_handler import load_dataset
from core.eda.summary import dataset_overview
from core.eda.correlations import compute_correlation
from core.eda.visualisation import plot_distribution, plot_heatmap

st.set_page_config(
    page_title="Explore Data",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Explore Data")

# Load uploaded dataset
uploaded = st.session_state.get("uploaded_file", None)

if not uploaded:
    st.warning("Please upload a dataset first from the 'Upload Dataset' page.")
    st.stop()

df = load_dataset()

# Convert to pandas for visualizations
pdf = df.to_pandas()

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Missing Values",
    "Distributions",
    "Correlations"
])

with tab1:
    st.subheader("Dataset Overview")
    overview = dataset_overview(df)
    st.write(overview)

    st.subheader("First 100 Rows")
    st.dataframe(pdf.head(100))

with tab2:
    st.subheader("Missing Value Summary")
    missing = df.null_count()
    missing_df = pd.DataFrame({
        "Column": missing.keys(),
        "Missing Values": missing.values()
    })
    st.dataframe(missing_df)

with tab3:
    st.subheader("Column Distributions")

    col = st.selectbox("Select Column", df.columns)
    fig = plot_distribution(pdf, col)
    if fig:
        st.pyplot(fig)
    else:
        st.warning("This column cannot be visualized.")

with tab4:
    st.subheader("Correlation Heatmap (Numeric Columns Only)")
    corr = compute_correlation(pdf)

    if corr is None:
        st.warning("No numeric columns found.")
    else:
        fig = plot_heatmap(corr)
        st.pyplot(fig)

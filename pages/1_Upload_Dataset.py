import streamlit as st
import polars as pl
from core.utils.file_handler import save_uploaded_file

st.title("Upload Dataset")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded:
    df, path = save_uploaded_file(uploaded)

    st.session_state["uploaded_df"] = df
    st.session_state["uploaded_path"] = path

    st.success("Dataset uploaded successfully!")
    st.write(df.head(50))

    st.info(f"Rows: {df.height()} | Columns: {len(df.columns)}")

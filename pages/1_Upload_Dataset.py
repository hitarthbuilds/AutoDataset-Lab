# pages/1_Upload_Dataset.py
import streamlit as st
import polars as pl
from core.utils.sessions import set_df
from core.utils.file_handler import save_uploaded_file

st.title("Upload Dataset")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # try to save the raw uploaded file
    try:
        saved_path = save_uploaded_file(uploaded_file)
    except Exception as e:
        st.error(f"Failed to save uploaded file: {e}")
        saved_path = None

    # read into polars
    try:
        df = pl.read_csv(uploaded_file)
        set_df(df)
        st.success("File uploaded successfully!")

        # NOTE: polars DataFrame properties: .height and .width (not funcs)
        st.info(f"Rows: {df.height} | Columns: {df.width}")

        st.subheader("Preview (First 100 Rows)")
        st.dataframe(df.head(100).to_pandas())

    except Exception as e:
        st.error(f"Error loading CSV: {e}")

import streamlit as st
from core.utils.file_handler import load_saved_file
import polars as pl

def get_df() -> pl.DataFrame:
    # If DF already in session
    if "uploaded_df" in st.session_state:
        return st.session_state["uploaded_df"]

    # If file is saved but df not loaded yet
    if "uploaded_path" in st.session_state:
        df = load_saved_file(st.session_state["uploaded_path"])
        if df is not None:
            st.session_state["uploaded_df"] = df
            return df
        else:
            st.error("Dataset file missing. Please upload again.")
            st.stop()

    # Nothing exists
    st.error("No dataset found. Please upload a CSV first.")
    st.stop()

import streamlit as st
import polars as pl
from core.utils.file_handler import load_saved_file, save_uploaded_file


def get_df():
    """
    Always returns a valid Polars DataFrame if uploaded,
    otherwise returns None.
    """
    # If a DF is already in session, use it
    if "uploaded_df" in st.session_state:
        return st.session_state["uploaded_df"]

    # If a path exists, load CSV from disk
    if "uploaded_path" in st.session_state:
        df = load_saved_file(st.session_state["uploaded_path"])
        if df is not None:
            st.session_state["uploaded_df"] = df
            return df

    return None


def set_df(df: pl.DataFrame):
    """
    Stores dataframe in session and keeps a temp CSV on disk.
    """
    st.session_state["uploaded_df"] = df

    # Save updated file to disk so other pages can reload it
    saved_path = save_uploaded_file(df)
    st.session_state["uploaded_path"] = saved_path

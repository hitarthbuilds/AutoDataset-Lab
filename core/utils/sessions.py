# core/utils/sessions.py
import streamlit as st
import pandas as pd


def set_df(df: pd.DataFrame):
    """Store the main dataframe in session state."""
    if isinstance(df, pd.DataFrame):
        st.session_state["df"] = df.copy()
    else:
        st.session_state["df"] = None


def get_df() -> pd.DataFrame:
    """Retrieve the main dataframe."""
    df = st.session_state.get("df")
    return df if isinstance(df, pd.DataFrame) else None


def set_reference_df(df: pd.DataFrame):
    """Store reference dataset (for drift)."""
    if isinstance(df, pd.DataFrame):
        st.session_state["reference_df"] = df.copy()
    else:
        st.session_state["reference_df"] = None


def get_reference_df() -> pd.DataFrame:
    """Retrieve reference dataset."""
    df = st.session_state.get("reference_df")
    return df if isinstance(df, pd.DataFrame) else None


def clear_eda_results():
    """Reset all EDA outputs so UI doesn't reuse stale values."""
    keys = [
        "analysis", "quality", "missing", "anomalies", "drift",
        "schema", "feature_importance", "recommendations",
        "eda_done", "visuals"
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]

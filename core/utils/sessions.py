import streamlit as st
import pandas as pd

def set_df(df: pd.DataFrame):
    """Store DataFrame safely in session state."""
    if isinstance(df, pd.DataFrame):
        st.session_state["df"] = df
    else:
        st.session_state["df"] = None

def get_df() -> pd.DataFrame:
    """Retrieve DataFrame with validation."""
    df = st.session_state.get("df")
    if isinstance(df, pd.DataFrame):
        return df
    return None

import streamlit as st

def set_df(df):
    st.session_state["df"] = df

def get_df():
    return st.session_state.get("df")

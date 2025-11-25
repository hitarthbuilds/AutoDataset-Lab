import streamlit as st
import pandas as pd
from core.nlp.explain_columns import explain_columns
from core.nlp.insights import generate_insights
from core.nlp.summarize import summarize_dataset
from core.utils.file_handler import load_dataset

st.set_page_config(
    page_title="AI Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Insights AI")

# Load dataset
uploaded_file = st.session_state.get("uploaded_file", None)

if uploaded_file is None:
    st.warning("Upload a dataset first from the 'Upload Dataset' page.")
    st.stop()

df = load_dataset("data/current.csv")

tab1, tab2, tab3 = st.tabs([
    "Explain Columns",
    "Dataset Summary",
    "AI Insights"
])

with tab1:
    st.subheader("Explain Dataset Columns")
    column_explanations = explain_columns(df)
    st.write(column_explanations)

with tab2:
    st.subheader("Summarize Dataset")
    summary_text = summarize_dataset(df)
    st.write(summary_text)

with tab3:
    st.subheader("AI Insights & Observations")
    insights = generate_insights(df)
    st.write(insights)

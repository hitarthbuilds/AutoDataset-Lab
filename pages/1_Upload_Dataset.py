import streamlit as st
import pandas as pd
from core.utils.sessions import set_df, clear_eda_results

def app():
    st.title("📁 Upload Dataset")
    st.write("Upload CSV file to start your project.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            set_df(df)
            clear_eda_results()   # IMPORTANT FIX

            st.success("Dataset loaded successfully!")
            st.dataframe(df.head())
            st.info(f"Shape: {df.shape}")

        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")

if __name__ == "__main__":
    app()

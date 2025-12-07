import streamlit as st
import pandas as pd
from core.utils.sessions import set_df

def app():
    st.title("📁 Upload Dataset")
    st.write("Drag & drop or browse CSV file")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            set_df(df)

            st.success("Dataset loaded successfully!")
            st.write("### Preview")
            st.dataframe(df.head())
            st.info(f"Shape: {df.shape}")

        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")

if __name__ == "__main__":
    app()

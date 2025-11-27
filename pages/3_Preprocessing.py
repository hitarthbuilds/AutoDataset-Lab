import streamlit as st
import polars as pl
import pandas as pd
from core.utils.sessions import get_df, set_df

def app():
    st.header("Preprocessing")

    df = get_df()
    if df is None:
        st.warning("Upload dataset first.")
        return

    pdf = df.to_pandas()
    st.subheader("Preview (Before Processing)")
    st.dataframe(pdf.head())

    st.sidebar.header("Options")
    drop_cols = st.sidebar.checkbox("Drop columns with > 50% missing")
    impute_method = st.sidebar.selectbox(
        "Impute numeric values",
        ["none", "mean", "median"]
    )

    run = st.sidebar.button("Apply")

    if run:
        temp = pdf.copy()

        # Drop high-missing columns
        if drop_cols:
            thresh = len(temp) * 0.5
            temp = temp.dropna(thresh=thresh, axis=1)

        # Impute
        if impute_method != "none":
            num = temp.select_dtypes(include=["number"])
            if not num.empty:
                if impute_method == "mean":
                    temp[num.columns] = num.fillna(num.mean())
                else:
                    temp[num.columns] = num.fillna(num.median())

        # Convert back to Polars
        new_df = pl.from_pandas(temp)
        set_df(new_df)

        st.success("Preprocessing applied.")
        st.dataframe(temp.head())

import streamlit as st
import pandas as pd
import polars as pl
import traceback
from core.preprocess.pipeline import apply_cleaning

from core.utils.sessions import get_df, set_df
from core.preprocess.pipeline import (
    apply_cleaning,
    build_preprocessing_pipeline,
    run_preprocessing_pipeline,
    save_pipeline
)

# ---------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------
def app():
    st.header("🛠️ Preprocessing")

    try:
        # -----------------------------------------
        # LOAD DATA
        # -----------------------------------------
        df_polars = get_df()
        if df_polars is None:
            st.warning("Upload a dataset first.")
            return

        df = df_polars.to_pandas()

        # -----------------------------------------
        # PREVIEW BEFORE PROCESSING
        # -----------------------------------------
        st.subheader("Preview Before Processing")
        st.dataframe(df.head())

        # -----------------------------------------
        # SIDEBAR OPTIONS
        # -----------------------------------------

        st.sidebar.title("Cleaning")
        clean_cols = st.sidebar.checkbox("Clean column names")
        drop_missing = st.sidebar.checkbox("Drop columns > 50% missing")
        drop_dups = st.sidebar.checkbox("Drop duplicate rows")

        st.sidebar.title("Imputation")
        numeric_strategy = st.sidebar.selectbox(
            "Numeric Imputation",
            ["mean", "median", "most_frequent"]
        )
        cat_strategy = st.sidebar.selectbox(
            "Categorical Imputation",
            ["most_frequent", "constant"]
        )

        st.sidebar.title("Encoding")
        encoder = st.sidebar.selectbox(
            "Categorical Encoding",
            ["onehot", "ordinal", "none"]
        )

        st.sidebar.title("Scaling")
        scaler = st.sidebar.selectbox(
            "Numeric Scaling",
            ["standard", "minmax", "none"]
        )

        # -----------------------------------------
        # RUN BUTTON
        # -----------------------------------------
        if st.sidebar.button("🚀 Run Preprocessing"):
            try:
                # ------------------------------
                # 1) Apply Cleaning
                # ------------------------------
                options = {
                    "clean_columns": clean_cols,
                    "drop_missing": drop_missing,
                    "drop_duplicates": drop_dups
                }

                cleaned_df = apply_cleaning(df, options)

                # ------------------------------
                # 2) Build Pipeline
                # ------------------------------
                pipeline = build_preprocessing_pipeline(
                    cleaned_df,
                    numeric_strategy,
                    cat_strategy,
                    encoder,
                    scaler
                )

                # ------------------------------
                # 3) Run Pipeline
                # ------------------------------
                processed_df = run_preprocessing_pipeline(cleaned_df, pipeline)

                # ------------------------------
                # 4) Save Pipeline
                # ------------------------------
                save_pipeline(pipeline)

                st.success("✔ Preprocessing completed! Pipeline saved.")

                # ------------------------------
                # 5) Show Output
                # ------------------------------
                st.subheader("Processed Preview")
                st.dataframe(processed_df.head())

                # ------------------------------
                # 6) Store Back to Session
                # ------------------------------
                set_df(pl.from_pandas(processed_df))

            except Exception as e:
                st.error("Something went wrong.")
                st.text(traceback.format_exc())

    except Exception as e:
        st.error("Critical failure in the Preprocessing page.")
        st.text(traceback.format_exc())


# Required by Streamlit multipage
if __name__ == "__main__":
    app()

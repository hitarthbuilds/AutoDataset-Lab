import streamlit as st
import polars as pl
from core.utils.file_handler import load_saved_file
from core.preprocess.clean import clean_data
from core.preprocess.encode import encode_features
from core.preprocess.scale import scale_features
from core.preprocess.impute import impute_missing

# Load DataFrame
if "uploaded_df" not in st.session_state:
    if "uploaded_path" in st.session_state:
        df = load_saved_file(st.session_state["uploaded_path"])
        st.session_state["uploaded_df"] = df
    else:
        st.error("No dataset found. Please upload a CSV first.")
        st.stop()

df = st.session_state["uploaded_df"]

st.title("Preprocessing")

st.subheader("Choose Preprocessing Steps")

do_clean = st.checkbox("Clean Data (remove duplicates, standardize text)")
do_impute = st.checkbox("Impute Missing Values")
do_encode = st.checkbox("Encode Categorical Features")
do_scale = st.checkbox("Scale Numeric Features")

if st.button("Apply Preprocessing"):
    final_df = df.clone()

    if do_clean:
        final_df = clean_data(final_df)

    if do_impute:
        final_df = impute_missing(final_df)

    if do_encode:
        final_df = encode_features(final_df)

    if do_scale:
        final_df = scale_features(final_df)

    st.session_state["uploaded_df"] = final_df

    st.success("Preprocessing Applied!")
    st.dataframe(final_df.head(50))

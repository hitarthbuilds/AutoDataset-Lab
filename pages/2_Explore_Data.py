import streamlit as st
import polars as pl

from core.utils.sessions import get_df
from core.eda.summary import dataset_overview
from core.eda.missing_value import missing_summary
from core.eda.correlations import compute_correlations
from core.eda.visualisation import plot_numeric_distributions

st.title("Explore Data")

df = get_df()

if df is None:
    st.error("No dataset found. Upload a CSV first.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Missing Values",
    "Distributions",
    "Correlations"
])

# -----------------------------------------------------------------------------
# TAB 1: OVERVIEW
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Dataset Overview")

    try:
        overview = dataset_overview(df)
        st.write(overview)
    except Exception as e:
        st.error(f"Error in dataset_overview(): {e}")

    st.subheader("Preview (First 100 Rows)")
    st.dataframe(df.head(100))


# -----------------------------------------------------------------------------
# TAB 2: MISSING VALUES
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Missing Value Summary")

    try:
        missing_df = missing_summary(df)
        st.dataframe(missing_df)
    except Exception as e:
        st.error(f"Error in missing_summary(): {e}")


# -----------------------------------------------------------------------------
# TAB 3: NUMERIC DISTRIBUTIONS
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Numeric Distributions")

    try:
        fig = plot_numeric_distributions(df)
        if fig:
            st.pyplot(fig)
        else:
            st.info("No numeric columns available.")
    except Exception as e:
       st.error(f"Error in compute_correlations(): {e}")

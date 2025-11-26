import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

st.set_page_config(
    page_title="Preprocessing",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Preprocessing")

# --------------------------------------------
# Load uploaded dataset from session state
# --------------------------------------------
if "uploaded_df" not in st.session_state:
    st.error("⚠️ No dataset found. Please upload a CSV file first.")
    st.stop()

df = st.session_state["uploaded_df"].copy()

# --------------------------------------------
# SIDEBAR OPTIONS
# --------------------------------------------
st.sidebar.header("Preprocessing Options")

# Missing Values
missing_strategy = st.sidebar.selectbox(
    "Missing Value Strategy",
    [
        "Do nothing",
        "Drop rows with missing values",
        "Drop columns with missing values",
        "Fill with mean",
        "Fill with median",
        "Fill with mode",
        "Fill with custom value"
    ]
)

custom_fill = None
if missing_strategy == "Fill with custom value":
    custom_fill = st.sidebar.text_input("Enter custom fill value", "")

# Encoding
encoding_strategy = st.sidebar.selectbox(
    "Categorical Encoding",
    ["Do nothing", "Label Encoding", "One-Hot Encoding"]
)

# Scaling
scaling_strategy = st.sidebar.selectbox(
    "Scaling Method",
    ["Do nothing", "Standard Scaler", "MinMax Scaler"]
)

# Apply button
if st.sidebar.button("Apply Preprocessing"):
    processed_df = df.copy()

    # ---------------------------
    # 1. Handle Missing Values
    # ---------------------------
    if missing_strategy == "Drop rows with missing values":
        processed_df = processed_df.dropna()

    elif missing_strategy == "Drop columns with missing values":
        processed_df = processed_df.dropna(axis=1)

    elif missing_strategy == "Fill with mean":
        processed_df = processed_df.fillna(processed_df.mean(numeric_only=True))

    elif missing_strategy == "Fill with median":
        processed_df = processed_df.fillna(processed_df.median(numeric_only=True))

    elif missing_strategy == "Fill with mode":
        processed_df = processed_df.fillna(processed_df.mode().iloc[0])

    elif missing_strategy == "Fill with custom value":
        processed_df = processed_df.fillna(custom_fill)

    # ---------------------------
    # 2. Encode Categorical Columns
    # ---------------------------
    categorical_cols = processed_df.select_dtypes(include=["object"]).columns.tolist()

    if encoding_strategy == "Label Encoding":
        le = LabelEncoder()
        for col in categorical_cols:
            try:
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))
            except:
                pass

    elif encoding_strategy == "One-Hot Encoding":
        processed_df = pd.get_dummies(processed_df, columns=categorical_cols)

    # ---------------------------
    # 3. Scale Numeric Columns
    # ---------------------------
    numeric_cols = processed_df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if scaling_strategy == "Standard Scaler":
        scaler = StandardScaler()
        processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])

    elif scaling_strategy == "MinMax Scaler":
        scaler = MinMaxScaler()
        processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])

    # Save to session_state
    st.session_state["processed_df"] = processed_df

    st.success("✨ Preprocessing applied successfully!")

    st.subheader("Preview After Preprocessing")
    st.dataframe(processed_df.head(100))

else:
    st.info("▶️ Configure settings in the sidebar, then click 'Apply Preprocessing'.")

    st.subheader("Preview Before Preprocessing")
    st.dataframe(df.head(100))

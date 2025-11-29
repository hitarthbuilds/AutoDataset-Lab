import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from core.preprocess.clean import (
    clean_column_names,
    drop_duplicates,
    drop_missing_threshold
)


# -----------------------------------------------------
# CLEANING
# -----------------------------------------------------
def apply_cleaning(df: pd.DataFrame, options: dict):
    """Apply cleaning options before ML pipeline."""
    df = df.copy()

    if options.get("clean_columns"):
        df = clean_column_names(df)

    if options.get("drop_missing"):
        df = drop_missing_threshold(df, 0.5)

    if options.get("drop_duplicates"):
        df = drop_duplicates(df)

    return df


# -----------------------------------------------------
# IMPUTERS / ENCODERS / SCALERS
# -----------------------------------------------------
def get_imputer(strategy, is_numeric):
    if is_numeric:
        return SimpleImputer(strategy=strategy)
    else:
        if strategy == "constant":
            return SimpleImputer(strategy="constant", fill_value="missing")
        return SimpleImputer(strategy=strategy)


def get_encoder(name):
    name = name.lower().strip()

    if name == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if name == "ordinal":
        return OrdinalEncoder()

    if name == "none":
        return None

    raise ValueError(f"Unknown encoder: {name}")


def get_scaler(name):
    name = name.lower().strip()

    if name == "standard":
        return StandardScaler()

    if name == "minmax":
        return MinMaxScaler()

    if name == "none":
        return None

    raise ValueError(f"Unknown scaler: {name}")


# -----------------------------------------------------
# PIPELINE BUILDER
# -----------------------------------------------------
def build_preprocessing_pipeline(
    df: pd.DataFrame,
    numeric_imputer_strategy="mean",
    categorical_imputer_strategy="most_frequent",
    encoder="onehot",
    scaler="standard"
):
    """Build ColumnTransformer pipeline using REAL columns from df."""

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    num_imputer = get_imputer(numeric_imputer_strategy, True)
    cat_imputer = get_imputer(categorical_imputer_strategy, False)
    enc = get_encoder(encoder)
    scale = get_scaler(scaler)

    # NUMERIC PIPELINE
    num_steps = [("imputer", num_imputer)]
    if scale is not None:
        num_steps.append(("scaler", scale))

    numeric_pipeline = Pipeline(num_steps)

    # CATEGORICAL PIPELINE
    cat_steps = [("imputer", cat_imputer)]
    if enc is not None:
        cat_steps.append(("encoder", enc))

    categorical_pipeline = Pipeline(cat_steps)

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop"
    )

    return transformer


# -----------------------------------------------------
# RUNNING THE PIPELINE
# -----------------------------------------------------
def run_preprocessing_pipeline(df: pd.DataFrame, pipeline):
    transformed = pipeline.fit_transform(df)
    return pd.DataFrame(transformed)


# -----------------------------------------------------
# PIPELINE SAVE / LOAD
# -----------------------------------------------------
def save_pipeline(pipeline, path="pipeline.pkl"):
    joblib.dump(pipeline, path)


def load_pipeline(path="pipeline.pkl"):
    return joblib.load(path)

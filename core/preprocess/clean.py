# core/preprocess/clean.py
"""
Cleaning utilities for preprocessing.
Work in pandas DataFrame space (convert polars -> pandas before using).
"""

from typing import Tuple, List
import pandas as pd
import re


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names: lowercase, strip, replace spaces & punctuation with underscore.
    """
    df = df.copy()
    cleaned = []
    for c in df.columns:
        name = str(c).lower().strip()
        name = re.sub(r"[^\w]+", "_", name).strip("_")
        cleaned.append(name)
    df.columns = cleaned
    return df


def drop_missing_threshold(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Drop columns with fraction of missing values > threshold.
    threshold is fraction (0..1).
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    df = df.copy()
    missing_fraction = df.isna().mean()
    keep = missing_fraction[missing_fraction <= threshold].index.tolist()
    return df[keep]


def drop_duplicates(df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
    """
    Drop duplicate rows.
    """
    df = df.copy()
    return df.drop_duplicates(keep=keep)


def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Return (numeric_columns, categorical_columns) using pandas dtypes heuristics.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    other_cols = df.columns.difference(numeric_cols).tolist()
    return numeric_cols, other_cols

# core/preprocess/impute.py
"""
Imputer factory: returns sklearn SimpleImputer configured for numeric or categorical.
"""

from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
from typing import Optional


def get_imputer(strategy: str = "mean", is_numeric: bool = True) -> TransformerMixin:
    """
    Return a fitted (unfitted) SimpleImputer instance configured by strategy.

    strategy:
      - numeric: "mean", "median", "most_frequent", "constant"
      - categorical: "most_frequent", "constant"
    """
    s = strategy.lower().strip()
    if is_numeric:
        if s not in {"mean", "median", "most_frequent", "constant"}:
            raise ValueError(f"Unknown numeric imputer strategy: {strategy}")
        if s == "constant":
            return SimpleImputer(strategy="constant", fill_value=0)
        return SimpleImputer(strategy=s)
    else:
        if s not in {"most_frequent", "constant"}:
            raise ValueError(f"Unknown categorical imputer strategy: {strategy}")
        if s == "constant":
            return SimpleImputer(strategy="constant", fill_value="missing")
        return SimpleImputer(strategy="most_frequent")

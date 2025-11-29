# core/preprocess/scale.py
"""
Scaler factory.
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.base import TransformerMixin


def get_scaler(name: str = "standard") -> TransformerMixin:
    """
    Return a scaler. name: "standard", "minmax", "none"
    Using FunctionTransformer as no-op for 'none'.
    """
    n = (name or "none").lower().strip()
    if n == "standard":
        return StandardScaler()
    if n == "minmax":
        return MinMaxScaler()
    if n in {"none", "passthrough", ""}:
        return FunctionTransformer(lambda x: x)
    raise ValueError(f"Unknown scaler name: {name}")

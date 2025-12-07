# core/eda/utils_shap.py
from __future__ import annotations
import io
import base64
import traceback
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# encode dataframe columns to numeric matrix for model/SHAP consumption
def encode_df_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode object/categorical columns and coerce numerics to floats.
    Returns a numeric DataFrame ready for training or SHAP.
    """
    out = df.copy()
    for c in out.columns:
        try:
            if pd.api.types.is_numeric_dtype(out[c].dtype):
                out[c] = pd.to_numeric(out[c], errors="coerce").astype(float).fillna(0.0)
            else:
                # convert categories and strings to integers via factorize
                vals, _ = pd.factorize(out[c].astype(str).fillna("__NA__"))
                out[c] = vals.astype(float)
        except Exception:
            # last resort: convert everything to string scores
            out[c] = out[c].astype(str).fillna("__NA__").apply(lambda s: float(abs(hash(s)) % 100000) / 100000.0)
    # ensure float dtype
    for c in out.columns:
        out[c] = out[c].astype(float)
    return out

def safe_shap_explainer(model, X_enc: pd.DataFrame):
    """
    Return a shap explainer in a defensive way.
    Prefer TreeExplainer for tree models; otherwise fallback to KernelExplainer.
    """
    if not _HAS_SHAP:
        raise RuntimeError("shap not installed")
    try:
        # Try TreeExplainer first (fast & accurate for tree ensembles)
        try:
            explainer = shap.TreeExplainer(model, X_enc)
            return explainer
        except Exception:
            # fallback to general Explainer
            try:
                explainer = shap.Explainer(model, X_enc)
                return explainer
            except Exception:
                # brute fallback: KernelExplainer (slow)
                if hasattr(model, "predict_proba"):
                    predict_fn = model.predict_proba
                else:
                    predict_fn = model.predict
                explainer = shap.KernelExplainer(predict_fn, X_enc.iloc[:50, :])
                return explainer
    except Exception as e:
        raise

def save_matplotlib_figure_to_png(fig=None) -> bytes:
    """
    Save a matplotlib figure into PNG bytes.
    If fig is None, captures current figure.
    """
    import io
    bio = io.BytesIO()
    try:
        if fig is None:
            fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(bio, format="png", dpi=150)
        plt.close(fig)
        bio.seek(0)
        return bio.getvalue()
    except Exception:
        try:
            plt.close('all')
        except Exception:
            pass
        return b""

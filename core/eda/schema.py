"""
schema.py
ENTERPRISE-GRADE SCHEMA PROFILER & VALIDATOR
--------------------------------------------------------------

Capabilities:
    - Infers a complete dataset schema (types, ranges, categories)
    - Validates schema drift across versions
    - Detects suspicious type changes (float → string, int → categorical, etc.)
    - Computes column entropy, cardinality tiers, numeric distributions
    - Flags non-standard values (mixed types, parsing failures)
    - Produces JSON-ready schema including:
        - column profile
        - constraints
        - recommended fixes
        - schema_diff against a reference schema

Used by:
    - analysts
    - ML pipelines
    - data governance teams
    - audit reports
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _entropy(x: pd.Series) -> float:
    try:
        probs = x.value_counts(normalize=True, dropna=True)
        return float(-(probs * np.log2(probs + 1e-12)).sum())
    except Exception:
        return float("nan")


def _dtype_name(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_categorical_dtype(series):
        return "categorical"
    return "string"


def _infer_constraints(series: pd.Series, dtype: str) -> Dict[str, Any]:
    """
    Produces constraints appropriate for dtype.
    Useful for validation pipelines.
    """
    if dtype in ("integer", "float"):
        return {
            "min": float(series.min()) if series.dropna().size else None,
            "max": float(series.max()) if series.dropna().size else None,
            "mean": float(series.mean()) if series.dropna().size else None,
            "std": float(series.std()) if series.dropna().size else None,
        }

    if dtype == "categorical" or dtype == "string":
        vc = series.astype(str).value_counts()
        return {
            "unique": int(series.nunique(dropna=True)),
            "top_values": vc.head(20).to_dict(),
            "entropy": _entropy(series.astype(str)),
        }

    if dtype == "datetime":
        vals = pd.to_datetime(series, errors="coerce")
        return {
            "min_date": str(vals.min()) if vals.dropna().size else None,
            "max_date": str(vals.max()) if vals.dropna().size else None,
        }

    if dtype == "boolean":
        return {
            "true_pct": float(series.mean()) if series.dropna().size else None
        }

    return {}


# ------------------------------------------------------------
# 1. SCHEMA INFERENCE
# ------------------------------------------------------------

def infer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Produces a full schema with column-level profiles:
        - dtype
        - missing %
        - cardinality
        - constraints
        - suspicious values if dtype is inconsistent
    """

    schema = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": {},
    }

    for col in df.columns:
        ser = df[col]
        dtype = _dtype_name(ser)

        # detect mixed types
        mixed_flag = False
        if dtype == "string":
            uniques = ser.dropna().astype(str)
            if uniques.size > 0:
                try:
                    # detect if numeric inside string columns
                    parsed = pd.to_numeric(uniques, errors="coerce")
                    if parsed.notna().sum() / len(parsed) > 0.5:
                        mixed_flag = True
                except Exception:
                    pass

        profile = {
            "dtype": dtype,
            "missing": float(round(ser.isna().mean() * 100, 2)),
            "cardinality": int(ser.nunique(dropna=True)),
            "entropy": float(_entropy(ser.astype(str))) if dtype in ("string", "categorical") else None,
            "constraints": _infer_constraints(ser, dtype),
            "mixed_type_detected": mixed_flag,
        }

        schema["columns"][col] = profile

    return schema


# ------------------------------------------------------------
# 2. SCHEMA VALIDATION & DIFFING
# ------------------------------------------------------------

def compare_schemas(schema_old: Dict[str, Any], schema_new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compares two schema dictionaries and highlights:
        - type changes
        - cardinality shifts
        - new or dropped columns
        - constraint changes
    """

    diff = {
        "added_columns": [],
        "removed_columns": [],
        "type_changes": [],
        "cardinality_changes": [],
        "constraint_changes": [],
    }

    old_cols = schema_old.get("columns", {})
    new_cols = schema_new.get("columns", {})

    # added / removed
    for c in new_cols:
        if c not in old_cols:
            diff["added_columns"].append(c)

    for c in old_cols:
        if c not in new_cols:
            diff["removed_columns"].append(c)

    # detailed comparisons
    for col, new_profile in new_cols.items():
        if col not in old_cols:
            continue
        old_profile = old_cols[col]

        # dtype change
        if new_profile["dtype"] != old_profile["dtype"]:
            diff["type_changes"].append({
                "column": col,
                "old": old_profile["dtype"],
                "new": new_profile["dtype"],
            })

        # cardinality movement
        if new_profile["cardinality"] != old_profile["cardinality"]:
            diff["cardinality_changes"].append({
                "column": col,
                "old": old_profile["cardinality"],
                "new": new_profile["cardinality"],
            })

        # constraint-level diffs
        old_const = old_profile.get("constraints", {})
        new_const = new_profile.get("constraints", {})

        for k in set(old_const.keys()).union(new_const.keys()):
            if old_const.get(k) != new_const.get(k):
                diff["constraint_changes"].append({
                    "column": col,
                    "constraint": k,
                    "old": old_const.get(k),
                    "new": new_const.get(k),
                })

    return diff


# ------------------------------------------------------------
# 3. VALIDATION ENGINE
# ------------------------------------------------------------

def validate_against_schema(df: pd.DataFrame, expected_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates a dataframe against an expected schema and produces:
        - violations
        - missing columns
        - extra columns
        - type mismatches
        - value constraint violations
    """

    inferred = infer_schema(df)
    diff = compare_schemas(expected_schema, inferred)

    violations = []

    # column existence
    for c in diff["removed_columns"]:
        violations.append(f"Column missing: {c}")

    for c in diff["added_columns"]:
        violations.append(f"Unexpected column: {c}")

    # type mismatches
    for t in diff["type_changes"]:
        violations.append(f"Type change in '{t['column']}': {t['old']} → {t['new']}")

    # cardinality explosions (important for model drift)
    for card in diff["cardinality_changes"]:
        if card["new"] > card["old"] * 2:
            violations.append(f"Cardinality explosion in {card['column']}: {card['old']} → {card['new']}")

    # constraints
    for con in diff["constraint_changes"]:
        violations.append(
            f"Constraint change in {con['column']} — {con['constraint']}: {con['old']} → {con['new']}"
        )

    return {
        "inferred_schema": inferred,
        "schema_diff": diff,
        "violations": violations,
        "is_valid": len(violations) == 0,
    }

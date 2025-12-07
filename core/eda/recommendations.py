"""
Enterprise-grade recommendation engine for AutoDataset-Lab.

Produces:
- Human-readable “What I found” summaries
- Actionable “What to do next”
- LLM-ready recommendation blocks
- Structured JSON for export
- Multi-section audit summaries

This module synthesizes outputs from:
missing.py, quality.py, anomalies.py, drift.py, schema.py,
feature_importance.py, sampling.py, analyze.py
"""

from __future__ import annotations
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _fmt_list(items: List[str], max_items: int = 10) -> str:
    """Format list for readable bullet outputs."""
    if not items:
        return "None"
    if len(items) <= max_items:
        return ", ".join(items)
    return ", ".join(items[:max_items]) + f" … (+{len(items) - max_items} more)"

def _percent(v: float) -> str:
    try:
        return f"{round(float(v)*100, 2)}%"
    except Exception:
        return "N/A"

def _safe(v):
    try:
        if isinstance(v, (int, float, np.floating, np.integer)):
            return float(v)
        return v
    except Exception:
        return v

# ---------------------------------------------------------------------
# Main Recommendation Engine
# ---------------------------------------------------------------------

def compute_recommendations(
    analysis: Dict[str, Any],
    quality: Dict[str, Any],
    missing: Dict[str, Any],
    anomalies: Dict[str, Any],
    drift: Dict[str, Any],
    schema: Dict[str, Any],
    feature_importance: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Combine all EDA signals into:
    - executive findings
    - recommended actions
    - LLM-ready prompt
    - structured JSON export
    """

    # -----------------------------
    # SECTION A: Executive Findings
    # -----------------------------
    rows = analysis.get("rows")
    cols = analysis.get("columns")

    # Missingness summary
    missing_cols = sorted(
        missing.get("per_column", {}).keys(),
        key=lambda c: missing["per_column"][c]["missing_percent"],
        reverse=True
    )[:10]

    # High-cardinality
    high_card = [
        c for c, info in analysis.get("global_stats", {}).get("cardinality", {}).items()
        if info > 1000
    ]

    # Constant columns
    constant_cols = [
        c for c, info in analysis.get("columns_overview", {}).items()
        if info.get("is_constant")
    ]

    # Schema drift warnings
    schema_warnings = schema.get("schema_warnings", [])

    # Drift signals
    drift_score = drift.get("summary", {}).get("dataset_drift_score")
    drift_cols = drift.get("summary", {}).get("flagged_columns", [])

    # Anomalies summary
    anomaly_methods = anomalies.get("methods", {})
    anomaly_total = sum([
        m.get("total_anomalies", 0)
        for m in anomaly_methods.values()
    ])

    # Feature importance
    top_features = feature_importance.get("top_features") or []

    executive_findings = [
        f"Dataset contains **{rows:,} rows** and **{cols:,} columns**.",
        f"Top missingness columns: {_fmt_list(missing_cols)}.",
        f"High-cardinality columns: {_fmt_list(high_card)}.",
        f"Constant columns detected: {_fmt_list(constant_cols)}.",
    ]

    if anomaly_total > 0:
        executive_findings.append(f"Detected **{anomaly_total} anomalies** via statistical methods.")
    if drift_score is not None:
        executive_findings.append(f"Dataset drift score: **{drift_score}**.")
        if drift_cols:
            executive_findings.append(f"Drift-flagged columns: {_fmt_list(drift_cols)}.")
    if schema_warnings:
        executive_findings.append(f"Schema compatibility warnings: {_fmt_list(schema_warnings)}.")
    if top_features:
        executive_findings.append(f"Top predictive features identified: {_fmt_list([t['feature'] for t in top_features])}.")

    # -----------------------------
    # SECTION B: Recommended Actions
    # -----------------------------

    recommended_actions = []

    # Missingness
    for c in missing_cols:
        pct = missing["per_column"][c]["missing_percent"]
        if pct > 0.3:
            recommended_actions.append(
                f"Column **{c}** has high missingness ({_percent(pct)}). Consider drop, imputation, "
                "or feature engineering (missing indicator)."
            )
        elif pct > 0.05:
            recommended_actions.append(
                f"Column **{c}** has moderate missingness ({_percent(pct)}). Consider median/mode imputation."
            )

    # High-cardinality
    for c in high_card:
        recommended_actions.append(
            f"High-cardinality feature **{c}** detected; consider hashing, frequency encoding, or target encoding."
        )

    # Constant columns
    for c in constant_cols:
        recommended_actions.append(
            f"Column **{c}** is constant; safe to remove."
        )

    # Drift
    if drift_score and drift_score > 30:
        recommended_actions.append(
            f"Dataset drift score is **{drift_score}**, investigate changes in: {_fmt_list(drift_cols)}."
        )

    # Anomalies
    if anomaly_total > 0:
        recommended_actions.append(
            f"Detected anomalies; consider filtering or using robust models for noisy inputs."
        )

    # Schema warnings
    for w in schema_warnings:
        recommended_actions.append(
            f"Schema warning: {w}"
        )

    # -----------------------------
    # SECTION C: LLM Prompt
    # -----------------------------

    llm_prompt = f"""
You are a senior data analyst. Given the following dataset findings, write:

1. A 2–4 paragraph executive summary  
2. A prioritized action checklist  
3. Model-readiness risks and next steps  

FINDINGS:
{chr(10).join(['• ' + f for f in executive_findings])}

ACTIONS:
{chr(10).join(['• ' + a for a in recommended_actions])}

Return a polished, business-ready report.
""".strip()

    # -----------------------------
    # SECTION D: JSON Export
    # -----------------------------

    export_json = {
        "executive_findings": executive_findings,
        "recommended_actions": recommended_actions,
        "statistics": {
            "rows": rows,
            "columns": cols,
            "missing_top": missing_cols,
            "high_cardinality": high_card,
            "constant_columns": constant_cols,
            "drift_score": drift_score,
            "drift_columns": drift_cols,
            "anomalies_total": anomaly_total,
            "schema_warnings": schema_warnings,
        },
        "llm_prompt": llm_prompt,
    }

    return export_json

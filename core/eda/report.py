# core/eda/report.py
"""
Enterprise audit report generator for AutoDataset-Lab.

Public functions:
- generate_audit_report_html(...) -> str
- export_audit_report(...) -> dict with paths (html, optional pdf, json, csvs)

Features:
- Multi-page HTML (print-optimized) with page breaks
- Embeds visuals (plotly/matplotlib) as base64 PNG
- Exports: HTML, optional PDF (weasyprint or pdfkit/wkhtmltopdf), JSON summary, CSV attachments
- Large-column-splitting so per-column sections paginate to many pages (20-30 pages possible)
"""

from __future__ import annotations
import os
import io
import json
import math
import base64
import tempfile
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Image export helpers from plotly (optional)
_has_kaleido = False
try:
    # plotly.io.to_image needs kaleido installed in many setups
    import plotly.io as pio
    _ = pio.to_image  # type: ignore
    _has_kaleido = True
except Exception:
    _has_kaleido = False

# Optional PDF engines
_has_weasy = False
try:
    import weasyprint  # type: ignore
    _has_weasy = True
except Exception:
    _has_weasy = False

_has_pdfkit = False
try:
    import pdfkit  # type: ignore
    _has_pdfkit = True
except Exception:
    _has_pdfkit = False

# -------------------------
# Helpers: safe JSON dumps
# -------------------------
def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, pd.Series):
        return o.dropna().tolist()
    if isinstance(o, pd.DataFrame):
        return o.head(200).to_dict(orient="records")
    return str(o)


def safe_json_dumps(obj: Any, **kwargs) -> str:
    return json.dumps(obj, default=_json_default, **kwargs)

# -------------------------
# Helpers: images -> base64
# -------------------------
def fig_to_base64_png(fig) -> Optional[str]:
    """
    Accepts a Plotly figure (or any object with a write_image / to_image) or a Matplotlib figure.
    Returns base64 PNG string (data:image/png;base64,...)
    """
    if fig is None:
        return None

    # Plotly figure
    try:
        # plotly: try pio.to_image (kaleido) then fig.to_image
        if _has_kaleido:
            png_bytes = pio.to_image(fig, format="png", scale=2)  # type: ignore
        else:
            # fallback: try fig.to_image
            png_bytes = fig.to_image(format="png", scale=2)  # type: ignore
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        pass

    # Matplotlib figure: support fig.savefig
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        pass

    return None

# -------------------------
# Helpers: dataframe -> html table (print friendly)
# -------------------------
def df_to_html_table(df: pd.DataFrame, max_rows: int = 50, table_id: str = "") -> str:
    safe = df.copy()
    # convert long objects safely
    for c in safe.columns:
        if safe[c].dtype == "object":
            safe[c] = safe[c].astype(str).str.slice(0, 200)
    html = safe.head(max_rows).to_html(classes="audit-table", index=False, border=0, escape=False)
    if table_id:
        html = html.replace("class=\"audit-table\"", f"id=\"{table_id}\" class=\"audit-table\"")
    return html

# -------------------------
# Report Template
# -------------------------
_BASE_CSS = """
/* Minimal print-friendly styling. Inline so PDF engines pick it up. */
body { font-family: "Inter", "Helvetica", Arial, sans-serif; color:#111; background:#fff; margin:0; padding:0; }
.container { width: 1024px; margin: 36px auto; padding: 24px; }
.header { text-align:left; border-bottom: 1px solid #ddd; padding-bottom:12px; margin-bottom:18px; }
.h1 { font-size:34px; margin: 0 0 6px 0; }
.h2 { font-size:22px; margin: 18px 0 6px 0; }
.kpi-row { display:flex; gap:24px; margin:14px 0 28px 0; }
.kpi { flex:1; padding:12px; border:1px solid #eee; border-radius:6px; background:#fafafa; text-align:center; }
.kpi .num { font-size:28px; font-weight:700; }
.section { page-break-inside: avoid; margin-bottom: 22px; }
.subsection { margin: 10px 0; }
.audit-table { width:100%; border-collapse: collapse; font-size:12px; }
.audit-table th, .audit-table td { padding:6px 8px; border:1px solid #eee; text-align:left; vertical-align: top; }
.img-block { text-align:center; margin: 12px 0 16px 0; }
.note { font-size:13px; color:#444; margin-top:6px; }
.page-break { page-break-after: always; }
.small { font-size:12px; color:#444; }
.code { background:#0f1720; color:#e6f1ff; padding:10px; border-radius:6px; font-family: monospace; white-space: pre-wrap; }
"""

_HTML_HEADER = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AutoDataset-Lab Audit Report</title>
<style> {css} </style>
</head>
<body>
"""

_HTML_FOOTER = """
</body>
</html>
"""

# -------------------------
# Primary: generate html (no external templating)
# -------------------------
def generate_audit_report_html(
    output_title: str,
    analysis: Dict[str, Any],
    quality: Dict[str, Any],
    missing: Dict[str, Any],
    anomalies: Dict[str, Any],
    drift: Dict[str, Any],
    schema: Dict[str, Any],
    feature_importance: Dict[str, Any],
    visuals: Dict[str, Any] = None,
    per_column_limit: int = 6,
) -> str:
    """
    Build a printable HTML audit. visuals is dict[str, figure] (plotly/mpl)
    per_column_limit: how many columns to include per page / block (controls pagination)
    """
    visuals = visuals or {}
    css = _BASE_CSS
    html_parts = [_HTML_HEADER.format(css=css)]
    html_parts.append('<div class="container">')
    # Header / cover
    rows = analysis.get("rows", "N/A")
    cols = analysis.get("columns", "N/A")
    html_parts.append('<div class="header">')
    html_parts.append(f'<div class="h1">{output_title}</div>')
    html_parts.append(f'<div class="small">Generated by AutoDataset-Lab</div>')
    html_parts.append('</div>')

    # KPIs
    html_parts.append('<div class="kpi-row">')
    html_parts.append(f'<div class="kpi"><div class="small">Rows</div><div class="num">{rows:,}</div></div>')
    html_parts.append(f'<div class="kpi"><div class="small">Columns</div><div class="num">{cols:,}</div></div>')
    dup_rows = quality.get("duplicates", {}).get("duplicate_rows", 0)
    html_parts.append(f'<div class="kpi"><div class="small">Duplicate rows</div><div class="num">{dup_rows}</div></div>')
    const_cols = sum(1 for c,info in analysis.get("columns_overview", {}).items() if info.get("is_constant"))
    html_parts.append(f'<div class="kpi"><div class="small">Constant cols</div><div class="num">{const_cols}</div></div>')
    html_parts.append('</div>')  # kpi-row

    # Executive summary block
    html_parts.append('<div class="section">')
    html_parts.append('<div class="h2">Executive summary</div>')
    # short bullets: use compute_recommendations structure if present in 'analysis' or quality
    # try to read analysis["executive_findings"] else fallback
    exec_findings = analysis.get("executive_findings") or analysis.get("summary_bullets") or []
    if isinstance(exec_findings, list) and exec_findings:
        html_parts.append('<div class="subsection">')
        html_parts.append('<ul>')
        for item in exec_findings:
            html_parts.append(f'<li class="small">{item}</li>')
        html_parts.append('</ul>')
        html_parts.append('</div>')
    else:
        # fallback textual chunk
        html_parts.append('<div class="note">No executive findings object passed in analysis; include one for richer report.</div>')
    html_parts.append('</div>')  # section

    # Numeric dashboards & tables
    html_parts.append('<div class="section">')
    html_parts.append('<div class="h2">Numerics & Data Quality (tables)</div>')

    # missing per-column table
    try:
        missing_df = pd.DataFrame([
            {"column": col, "missing_count": info.get("missing_count", 0), "missing_percent": info.get("missing_percent", 0)}
            for col, info in (missing.get("per_column") or {}).items()
        ])
        if not missing_df.empty:
            html_parts.append('<div class="subsection"><div class="small">Missingness (per-column numeric)</div>')
            html_parts.append(df_to_html_table(missing_df.sort_values("missing_percent", ascending=False), max_rows=200, table_id="missingness"))
            html_parts.append('</div>')
    except Exception:
        html_parts.append('<div class="note">Missingness table generation failed.</div>')

    # categorical summary (if present)
    cat_summary = analysis.get("categorical_summary")
    if cat_summary:
        try:
            cat_df = pd.DataFrame(cat_summary)
            html_parts.append('<div class="subsection"><div class="small">Categorical summary</div>')
            html_parts.append(df_to_html_table(cat_df))
            html_parts.append('</div>')
        except Exception:
            pass

    # data_quality diagnostics table
    try:
        dq_rows = []
        for col, info in (analysis.get("columns_overview") or {}).items():
            dq_rows.append({
                "column": col,
                "is_constant": bool(info.get("is_constant")),
                "infinities": int(info.get("infinities") or 0),
                "negatives": int(info.get("negatives") or 0),
                "cardinality": int(info.get("cardinality") or 0)
            })
        dq_df = pd.DataFrame(dq_rows)
        if not dq_df.empty:
            html_parts.append('<div class="subsection"><div class="small">Data quality diagnostics</div>')
            html_parts.append(df_to_html_table(dq_df.head(200)))
            html_parts.append('</div>')
    except Exception:
        pass

    html_parts.append('</div>')  # section

    # Anomalies & Drift numeric summaries
    html_parts.append('<div class="section">')
    html_parts.append('<div class="h2">Anomalies & Drift — numeric summaries</div>')
    # anomalies numeric summary
    if anomalies and anomalies.get("summary"):
        try:
            an_df = pd.DataFrame(anomalies.get("summary", []))
            html_parts.append('<div class="subsection"><div class="small">Anomaly summary</div>')
            html_parts.append(df_to_html_table(an_df))
            html_parts.append('</div>')
        except Exception:
            html_parts.append('<div class="note">Anomalies summary not available.</div>')
    else:
        html_parts.append('<div class="note">No anomaly numeric summary available.</div>')
    # drift numeric summary
    if drift and drift.get("numeric_summary"):
        try:
            dr_df = pd.DataFrame(drift.get("numeric_summary", []))
            html_parts.append('<div class="subsection"><div class="small">Drift numeric summary</div>')
            html_parts.append(df_to_html_table(dr_df))
            html_parts.append('</div>')
        except Exception:
            html_parts.append('<div class="note">Drift numeric summary not available.</div>')
    else:
        html_parts.append('<div class="note">No drift numeric summary available.</div>')
    html_parts.append('</div>')  # section

    # Visual Panels (each image on own printable block)
    if visuals:
        html_parts.append('<div class="section">')
        html_parts.append('<div class="h2">Visual panels</div>')
        for name, fig in visuals.items():
            html_parts.append('<div class="subsection">')
            html_parts.append(f'<div class="small">{name}</div>')
            try:
                img_b64 = fig_to_base64_png(fig)
            except Exception:
                img_b64 = None
            if img_b64:
                html_parts.append(f'<div class="img-block"><img src="{img_b64}" style="max-width:100%; border:1px solid #eee;"/></div>')
            else:
                html_parts.append('<div class="note">Visual export failed (no kaleido / matplotlib support). Display may be missing in PDF.</div>')
            html_parts.append('</div>')
            html_parts.append('<div class="page-break"></div>')
        html_parts.append('</div>')  # section

    # Per-column drilldown pages (split into chunks to produce multiple pages)
    cols_list = list((analysis.get("columns_overview") or {}).keys())
    if cols_list:
        chunk_size = max(1, per_column_limit)
        chunks = [cols_list[i:i+chunk_size] for i in range(0, len(cols_list), chunk_size)]
        for chunk in chunks:
            html_parts.append('<div class="section">')
            html_parts.append('<div class="h2">Per-column diagnostics</div>')
            rows = []
            for c in chunk:
                info = (analysis.get("columns_overview") or {}).get(c, {})
                rows.append({
                    "column": c,
                    "dtype": info.get("dtype", ""),
                    "missing_percent": info.get("missing_percent", 0),
                    "unique": info.get("unique", ""),
                    "is_constant": bool(info.get("is_constant", False))
                })
            html_parts.append(df_to_html_table(pd.DataFrame(rows)))
            html_parts.append('</div>')
            html_parts.append('<div class="page-break"></div>')

    # Recommendations block if analysis contains one
    recs = analysis.get("recommended_actions") or analysis.get("recommendations") or []
    if recs:
        html_parts.append('<div class="section">')
        html_parts.append('<div class="h2">Recommendations (actionable)</div>')
        html_parts.append('<div class="subsection"><ul>')
        for r in recs:
            html_parts.append(f'<li class="small">{r}</li>')
        html_parts.append('</ul></div>')
        html_parts.append('</div>')

    # Append LLM prompt text if present
    llm = analysis.get("llm_prompt") or ""
    if llm:
        html_parts.append('<div class="section">')
        html_parts.append('<div class="h2">LLM Prompt (copyable)</div>')
        html_parts.append('<div class="code">')
        html_parts.append(llm.replace("&", "&amp;").replace("<", "&lt;"))
        html_parts.append('</div>')
        html_parts.append('</div>')

    # Footer and end
    html_parts.append('</div>')  # container
    html_parts.append(_HTML_FOOTER)
    html = "\n".join(html_parts)
    return html

# -------------------------
# Export wrapper: html -> file, optional pdf
# -------------------------
def export_audit_report(
    output_path: str,
    output_title: str,
    analysis: Dict[str, Any],
    quality: Dict[str, Any],
    missing: Dict[str, Any],
    anomalies: Dict[str, Any],
    drift: Dict[str, Any],
    schema: Dict[str, Any],
    feature_importance: Dict[str, Any],
    visuals: Dict[str, Any] = None,
    generate_pdf: bool = True,
    per_column_limit: int = 6,
) -> Dict[str, Any]:
    """
    Generate files:
    - output_path.html (required)
    - optionally output_path.pdf (if weasyprint or pdfkit/wkhtmltopdf present)
    - output_path.json (structured export of the combined results)
    - CSVs for missingness / quality if present

    Returns dict with file paths and status flags.
    """
    out = {"html": None, "pdf": None, "json": None, "csv": [], "warnings": []}
    base = os.path.splitext(output_path)[0]
    html_path = base + ".html"
    json_path = base + ".json"

    # Build a combined object for JSON export (safe)
    combined = {
        "analysis": analysis,
        "quality": quality,
        "missing": missing,
        "anomalies": anomalies,
        "drift": drift,
        "schema": schema,
        "feature_importance": feature_importance,
    }

    # 1) HTML
    html = generate_audit_report_html(
        output_title=output_title,
        analysis=analysis,
        quality=quality,
        missing=missing,
        anomalies=anomalies,
        drift=drift,
        schema=schema,
        feature_importance=feature_importance,
        visuals=visuals,
        per_column_limit=per_column_limit,
    )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    out["html"] = html_path

    # 2) JSON
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(safe_json_dumps(combined, indent=2))
    out["json"] = json_path

    # 3) CSV exports for missingness / quality tables when present
    try:
        if missing and missing.get("per_column"):
            missing_df = pd.DataFrame([
                {"column": c, **info} for c, info in missing["per_column"].items()
            ])
            csvp = base + ".missingness.csv"
            missing_df.to_csv(csvp, index=False)
            out["csv"].append(csvp)
    except Exception as e:
        out["warnings"].append(f"failed to export missingness csv: {e}")

    try:
        if analysis and analysis.get("columns_overview"):
            dq_df = pd.DataFrame([
                {"column": c, **info} for c, info in analysis["columns_overview"].items()
            ])
            csvp = base + ".data_quality.csv"
            dq_df.to_csv(csvp, index=False)
            out["csv"].append(csvp)
    except Exception as e:
        out["warnings"].append(f"failed to export data_quality csv: {e}")

    # 4) PDF generation (optional)
    pdf_written = False
    pdf_path = base + ".pdf"
    if generate_pdf:
        if _has_weasy:
            try:
                weasyprint.HTML(string=html).write_pdf(pdf_path)
                out["pdf"] = pdf_path
                pdf_written = True
            except Exception as e:
                out["warnings"].append(f"WeasyPrint PDF generation failed: {e}")
        elif _has_pdfkit:
            try:
                # user must have wkhtmltopdf on PATH
                pdfkit.from_file(html_path, pdf_path)
                out["pdf"] = pdf_path
                pdf_written = True
            except Exception as e:
                out["warnings"].append(f"pdfkit/wkhtmltopdf generation failed: {e}")
        else:
            out["warnings"].append("No PDF engine available (weasyprint or pdfkit required). HTML saved.")

    # 5) Ensure visuals exported into separate files if user wants them (not required)
    # note: images are embedded in HTML already

    return out

# -------------------------
# CLI demo (for dev / smoke tests)
# -------------------------
if __name__ == "__main__":
    # quick smoke demo with small fake objects
    demo_analysis = {
        "rows": 1234,
        "columns": 12,
        "columns_overview": {
            "A": {"dtype": "float", "missing_percent": 0.0, "unique": 100, "is_constant": False},
            "B": {"dtype": "object", "missing_percent": 0.5, "unique": 4, "is_constant": False},
            "C": {"dtype": "int", "missing_percent": 0.0, "unique": 1, "is_constant": True},
        },
        "executive_findings": [
            "Dataset shape: 1,234 × 12",
            "Top missingness columns: B (50%)",
            "1 constant column: C"
        ],
        "recommended_actions": [
            "Drop column C (constant).",
            "Impute column B using median or mode depending on semantic type."
        ]
    }

    demo_quality = {"duplicates": {"duplicate_rows": 2}}
    demo_missing = {"per_column": {"A": {"missing_count": 0, "missing_percent": 0.0}, "B": {"missing_count": 617, "missing_percent": 0.5}}}
    demo_anomalies = {"summary": []}
    demo_drift = {}
    demo_schema = {"schema_warnings": []}
    demo_feature_importance = {"top_features": [{"feature": "A", "score": 0.52}]}

    out = export_audit_report(
        output_path="audit_demo",
        output_title="Audit Demo - AutoDataset-Lab",
        analysis=demo_analysis,
        quality=demo_quality,
        missing=demo_missing,
        anomalies=demo_anomalies,
        drift=demo_drift,
        schema=demo_schema,
        feature_importance=demo_feature_importance,
        visuals=None,
        generate_pdf=False,
    )
    print("Generated:", out)

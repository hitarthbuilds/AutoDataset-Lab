# core/eda/report_helpers.py
from __future__ import annotations
import io
import textwrap
from typing import Dict, Any

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

PAGE_W, PAGE_H = A4

def _safe_text_lines(text: str, width=90):
    if text is None:
        return []
    return textwrap.wrap(str(text), width=width)

def build_multipage_pdf_with_images(analysis: Dict[str, Any], images: Dict[str, bytes]) -> bytes:
    """
    Build a multipage audit PDF embedding images provided as bytes.
    - analysis: EDA JSON-style dict
    - images: mapping "name" -> png bytes
    Returns PDF bytes
    """
    bio = io.BytesIO()
    c = canvas.Canvas(bio, pagesize=A4)
    margin = 40
    y = PAGE_H - 60
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, "EDA Audit Report")
    c.setFont("Helvetica", 10)
    y -= 30
    c.drawString(margin, y, f"Rows: {analysis.get('rows', '?')}   Columns: {analysis.get('cols', '?')}")
    y -= 20
    c.line(margin, y, PAGE_W - margin, y)
    y -= 30

    # Executive summary
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Executive summary")
    y -= 18
    c.setFont("Helvetica", 10)
    summary = analysis.get("summary", analysis.get("feature_importance", {}).get("meta", {}).get("warning", "Auto-generated EDA"))
    for L in _safe_text_lines(summary, width=110):
        if y < 120:
            c.showPage(); y = PAGE_H - 60
        c.drawString(margin, y, L)
        y -= 12
    y -= 6

    # insert each image on its own page (if too big)
    for name, png in images.items():
        try:
            c.showPage()
            y = PAGE_H - 60
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y, f"Figure: {name}")
            y -= 20
            img = ImageReader(io.BytesIO(png))
            # compute fit
            iw, ih = img.getSize()
            max_w = PAGE_W - 2*margin
            max_h = PAGE_H - 2*margin - 40
            scale = min(max_w / iw, max_h / ih, 1.0)
            draw_w = iw * scale
            draw_h = ih * scale
            x = (PAGE_W - draw_w) / 2
            c.drawImage(img, x, (PAGE_H - margin - draw_h), width=draw_w, height=draw_h)
        except Exception:
            # non-fatal, skip
            continue

    # final: write analysis JSON summary page
    c.showPage()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, PAGE_H - 60, "Appendix: Analysis (key entries)")
    c.setFont("Helvetica", 9)
    y = PAGE_H - 80
    try:
        key_items = {
            "rows": analysis.get("rows"),
            "cols": analysis.get("cols"),
            "missing_sample": list(analysis.get("missing", {}).items())[:10],
            "top_features": analysis.get("feature_importance", {}).get("aggregated", {}).get("top_features", [])[:20]
        }
        import json
        summary_text = json.dumps(key_items, default=str, indent=2)
        for L in summary_text.splitlines():
            if y < 80:
                c.showPage(); y = PAGE_H - 60
            c.drawString(margin, y, L[:200])
            y -= 12
    except Exception:
        pass

    c.save()
    bio.seek(0)
    return bio.getvalue()

"""
report.py  —  PDF REPORT GENERATION
===================================
Builds a PDF report of fire/smoke detections from the CSV log, filtered by a
date-time range and (optionally) a single camera.

Used by the dashboard's Export button via web_app.py, but also runnable on its
own for testing:

    python report.py                 # full report -> report.pdf
    python report.py 2026-06-01 2026-06-30 "Floor-1 Laptop-cam0"

Public function:
    build_report(start=None, end=None, camera=None) -> BytesIO (PDF bytes)

start/end are datetime objects (or None = unbounded).
camera is a camera name string (or None / "all" = every camera).
"""

import csv
import os
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle)

import config

# Theme colors (match the dashboard)
FIRE = colors.HexColor("#c0392b")
SMOKE = colors.HexColor("#e8a33d")
DARK = colors.HexColor("#1a1a1a")
GREY = colors.HexColor("#595959")
LINE = colors.HexColor("#cccccc")
ORG = getattr(config, "ORG_NAME", "Intrino Robotics & Technologies")


# ==========================
# Read + filter the CSV
# ==========================
def _read_rows(start=None, end=None, camera=None):
    rows = []
    if not os.path.exists(config.CSV_FILE):
        return rows

    cam_filter = None if (camera in (None, "all", "All", "")) else camera

    with open(config.CSV_FILE, newline="") as f:
        for r in csv.DictReader(f):
            # Build a datetime from the Date + Time columns
            try:
                dt = datetime.strptime(f"{r['Date']} {r['Time']}", "%Y-%m-%d %H:%M:%S")
            except (ValueError, KeyError):
                continue
            if start and dt < start:
                continue
            if end and dt > end:
                continue
            if cam_filter and r.get("Camera") != cam_filter:
                continue
            rows.append(r)
    return rows


# ==========================
# Build the PDF
# ==========================
def build_report(start=None, end=None, camera=None):
    rows = _read_rows(start, end, camera)

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm,
                            title="Fire & Smoke Detection Report")

    styles = getSampleStyleSheet()
    h_title = ParagraphStyle("t", parent=styles["Title"], fontName="Helvetica-Bold",
                             fontSize=20, textColor=DARK, spaceAfter=2)
    h_sub = ParagraphStyle("s", parent=styles["Normal"], fontName="Helvetica",
                           fontSize=10, textColor=GREY, spaceAfter=12)
    h_sec = ParagraphStyle("sec", parent=styles["Normal"], fontName="Helvetica-Bold",
                           fontSize=12, textColor=FIRE, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("b", parent=styles["Normal"], fontName="Helvetica",
                          fontSize=10, textColor=DARK, leading=15)

    el = []

    # ---- Header ----
    el.append(Paragraph("Fire &amp; Smoke Detection Report", h_title))
    el.append(Paragraph(f"{ORG} &nbsp;&bull;&nbsp; Generated "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", h_sub))

    # ---- Filters applied ----
    fr = start.strftime("%Y-%m-%d %H:%M") if start else "Beginning"
    to = end.strftime("%Y-%m-%d %H:%M") if end else "Now"
    cam_label = camera if (camera and camera not in ("all", "All", "")) else "All cameras"
    filt = [
        ["Date / time range:", f"{fr}  to  {to}"],
        ["Camera:", cam_label],
        ["Total detections:", str(len(rows))],
    ]
    ft = Table(filt, colWidths=[40 * mm, 130 * mm])
    ft.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    el.append(ft)

    # ---- Summary ----
    fire_n = sum(1 for r in rows if r.get("Detection", "").lower() == "fire")
    smoke_n = sum(1 for r in rows if r.get("Detection", "").lower() == "smoke")
    by_cam = {}
    for r in rows:
        by_cam[r.get("Camera", "?")] = by_cam.get(r.get("Camera", "?"), 0) + 1

    el.append(Paragraph("Summary", h_sec))
    sm = [["Fire detections", str(fire_n)], ["Smoke detections", str(smoke_n)]]
    for cam, n in sorted(by_cam.items()):
        sm.append([f"  {cam}", str(n)])
    st = Table(sm, colWidths=[130 * mm, 40 * mm])
    st.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
    ]))
    el.append(st)

    # ---- Detail table ----
    el.append(Paragraph("Detection Log", h_sec))
    if not rows:
        el.append(Paragraph("No detections in the selected range.", body))
    else:
        data = [["Date", "Time", "Camera", "Type", "Conf."]]
        for r in rows:
            data.append([r.get("Date", ""), r.get("Time", ""), r.get("Camera", ""),
                         r.get("Detection", "").upper(), r.get("Confidence", "")])
        dt = Table(data, colWidths=[24 * mm, 22 * mm, 70 * mm, 26 * mm, 18 * mm],
                   repeatRows=1)
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, LINE),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]
        # color the Type cell per row
        for i, r in enumerate(rows, start=1):
            c = FIRE if r.get("Detection", "").lower() == "fire" else SMOKE
            style.append(("TEXTCOLOR", (3, i), (3, i), c))
            style.append(("FONTNAME", (3, i), (3, i), "Helvetica-Bold"))
        dt.setStyle(TableStyle(style))
        el.append(dt)

    # ---- Footer with page numbers ----
    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GREY)
        canvas.drawCentredString(
            A4[0] / 2, 10 * mm,
            f"{ORG}  -  Confidential  -  Fire & Smoke Detection Report  -  Page {d.page}")
        canvas.restoreState()

    doc.build(el, onFirstPage=footer, onLaterPages=footer)
    buf.seek(0)
    return buf


# ==========================
# CLI test
# ==========================
if __name__ == "__main__":
    import sys
    s = e = c = None
    if len(sys.argv) > 1:
        s = datetime.strptime(sys.argv[1], "%Y-%m-%d")
    if len(sys.argv) > 2:
        e = datetime.strptime(sys.argv[2] + " 23:59:59", "%Y-%m-%d %H:%M:%S")
    if len(sys.argv) > 3:
        c = sys.argv[3]
    pdf = build_report(s, e, c)
    with open("report.pdf", "wb") as f:
        f.write(pdf.read())
    print("Wrote report.pdf")
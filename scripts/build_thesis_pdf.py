#!/usr/bin/env python3
"""Build the citeable CONFLUENCE thesis-findings PDF (reportlab).

Output: evidence/thesis.pdf (site root on the evidence Vercel project).
Google Scholar requires searchable text, a large title, author line, date,
and a numbered References section. This PDF is research-only.

    python scripts/build_thesis_pdf.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

REPO = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPO / "docs" / "manuscript" / "thesis_01_confluence_onco.md"
OUTPUT = REPO / "evidence" / "thesis.pdf"

TITLE = (
    "CONFLUENCE × OnCo: An Evidence-Gated Dynamical Framework for Integrating "
    "Oncology Knowledge Graphs with Adaptive Cancer-State Models"
)
AUTHOR = "Kelechi Emeka Ogbonna"
DATE_DISPLAY = "20 September 2026"
INSTITUTION = (
    "Independent computational research / Project Confluence "
    "(GitHub cloudynirvana)"
)
HTML_URL = "https://confluence-research.vercel.app/thesis"
PDF_URL = "https://confluence-research.vercel.app/thesis.pdf"
GITHUB = "https://github.com/cloudynirvana/project-confluence"

ABSTRACT = (
    "Cancer research now produces knowledge graphs, multi-omic assays and "
    "dynamical simulators in parallel [10,53,61]. The failure mode is collapsing "
    "those layers. This study asks whether a provenance-controlled pipeline can "
    "keep the layers apart while still allowing testable predictions. Scientific "
    "success is a staged chain — traceability, mathematical validity, "
    "identifiability, out-of-sample prediction, experimental falsification — "
    "not disease eradication [43-52]. PR #9 is architectural evidence for the "
    "gates, not therapeutic efficacy [11,56]. GLOBOCAN 2024 estimates, published "
    "2026, report about 20.6 million diagnoses and 9.8 million deaths; that "
    "statistic is context, not a CONFLUENCE parameter [1,2,64]."
)

DISCLAIMER = (
    "RESEARCH TECHNICAL REPORT. Project Confluence is a computational research "
    "framework. It is not a medical device, not a clinical decision-support "
    "system, not a diagnostic or therapeutic product, and not a protocol. "
    "This document makes no cure claim, no dosing recommendation, and no "
    "claim of patient benefit. Simulated trajectories are not patient outcomes. "
    "Current results are computational unless an external experiment is cited."
)

ONCO_LINE = (
    "OnCo data (onco.cc) is cited under CC BY-NC 4.0; commercial use needs a "
    "licence. CONFLUENCE software is MIT. This PDF has no DOI."
)

SERIF = "ThesisSerif"
SERIF_B = "ThesisSerif-Bold"
SERIF_I = "ThesisSerif-Italic"
SERIF_BI = "ThesisSerif-BoldItalic"
SANS = "ThesisSans"
SANS_B = "ThesisSans-Bold"

NAVY = HexColor("#1b3a4b")
INK = HexColor("#1b1a17")
MUTED = HexColor("#5c574e")
GOLD = HexColor("#8a6a2f")
ROSE_BG = HexColor("#f8eeeb")
ROSE = HexColor("#8a3a32")
LINE = HexColor("#d9d1c3")
STAMP_BG = HexColor("#d4a054")


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def register_fonts() -> None:
    serif = _first_existing(
        [
            Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
        ]
    )
    serif_b = _first_existing(
        [
            Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
        ]
    )
    serif_i = _first_existing(
        [
            Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
        ]
    )
    serif_bi = _first_existing(
        [
            Path("/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
        ]
    )
    sans = _first_existing(
        [
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )
    sans_b = _first_existing(
        [
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ]
    )
    if not all([serif, serif_b, serif_i, serif_bi, sans, sans_b]):
        # Type 1 core fonts: still searchable; Scholar prefers them to Type 3.
        global SERIF, SERIF_B, SERIF_I, SERIF_BI, SANS, SANS_B
        SERIF, SERIF_B, SERIF_I, SERIF_BI = "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"
        SANS, SANS_B = "Helvetica", "Helvetica-Bold"
        return
    pdfmetrics.registerFont(TTFont(SERIF, serif))
    pdfmetrics.registerFont(TTFont(SERIF_B, serif_b))
    pdfmetrics.registerFont(TTFont(SERIF_I, serif_i))
    pdfmetrics.registerFont(TTFont(SERIF_BI, serif_bi))
    pdfmetrics.registerFont(TTFont(SANS, sans))
    pdfmetrics.registerFont(TTFont(SANS_B, sans_b))
    pdfmetrics.registerFontFamily(
        SERIF,
        normal=SERIF,
        bold=SERIF_B,
        italic=SERIF_I,
        boldItalic=SERIF_BI,
    )


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PaperTitle",
            parent=base["Normal"],
            fontName=SERIF_B,
            fontSize=24,
            leading=28,
            alignment=TA_LEFT,
            textColor=NAVY,
            spaceAfter=12,
        ),
        "author": ParagraphStyle(
            "PaperAuthor",
            parent=base["Normal"],
            fontName=SERIF,
            fontSize=16,
            leading=20,
            textColor=NAVY,
            spaceAfter=4,
        ),
        "meta": ParagraphStyle(
            "PaperMeta",
            parent=base["Normal"],
            fontName=SANS,
            fontSize=9.5,
            leading=13,
            textColor=MUTED,
            spaceAfter=3,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Normal"],
            fontName=SERIF_B,
            fontSize=13,
            leading=17,
            textColor=NAVY,
            spaceBefore=16,
            spaceAfter=8,
            borderPadding=2,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Normal"],
            fontName=SERIF_B,
            fontSize=11.5,
            leading=15,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontName=SERIF,
            fontSize=10.5,
            leading=14.5,
            alignment=TA_JUSTIFY,
            textColor=INK,
            spaceAfter=8,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer",
            parent=base["Normal"],
            fontName=SANS,
            fontSize=9,
            leading=12.5,
            textColor=HexColor("#3a201c"),
        ),
        "ref": ParagraphStyle(
            "Ref",
            parent=base["Normal"],
            fontName=SERIF,
            fontSize=9.5,
            leading=13,
            textColor=INK,
            leftIndent=16,
            firstLineIndent=-16,
            spaceAfter=5,
        ),
    }


def md_inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(.+?)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<font face='%s' size='8'>\1</font>" % SANS, text)
    text = re.sub(r"\[(\d+(?:[-,]\d+)*)\]", r"[\1]", text)
    return text


def parse_manuscript(raw: str) -> list[tuple[str, str]]:
    """Return (kind, text) blocks. kind in title, meta, h1, h2, p, refs_header, ref."""
    lines = raw.replace("\r\n", "\n").split("\n")
    blocks: list[tuple[str, str]] = []
    buf: list[str] = []
    in_refs = False

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        text = " ".join(part.strip() for part in buf if part.strip())
        buf = []
        if not text:
            return
        if in_refs and re.match(r"^\d+\.\s", text):
            blocks.append(("ref", text))
        else:
            blocks.append(("p", text))

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not blocks:
            # Manuscript title is a working subtitle, not the citeable PDF title.
            blocks.append(("subtitle", stripped[2:].strip()))
            continue
        if stripped.startswith("**") and ":**" in stripped:
            flush()
            blocks.append(("meta", stripped.replace("**", "")))
            continue
        if stripped == "## 8. REFERENCES" or stripped.lower() == "## references":
            flush()
            in_refs = True
            blocks.append(("h1", "References"))
            continue
        if stripped.startswith("## "):
            flush()
            in_refs = False
            blocks.append(("h1", stripped[3:].strip()))
            continue
        if stripped.startswith("### "):
            flush()
            blocks.append(("h2", stripped[4:].strip()))
            continue
        if re.match(r"^\d+\.\s", stripped):
            flush()
            if in_refs:
                blocks.append(("ref", stripped))
            else:
                blocks.append(("p", stripped))
            continue
        if stripped.startswith("|"):
            flush()
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= set("-: ") and c for c in cells):
                continue
            blocks.append(("p", " — ".join(cells)))
            continue
        if stripped in {"---", "***"} or stripped.startswith("<!--"):
            flush()
            continue
        if not stripped:
            flush()
            continue
        buf.append(stripped)
    flush()
    return blocks


def disclaimer_table(s) -> Table:
    inner = Paragraph(DISCLAIMER, s["disclaimer"])
    attr = Paragraph(ONCO_LINE, s["disclaimer"])
    data = [[inner], [attr]]
    table = Table(data, colWidths=[6.5 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), ROSE_BG),
                ("BOX", (0, 0), (-1, -1), 0.6, ROSE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
                ("TOPPADDING", (0, 1), (-1, 1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def add_page_bits(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(STAMP_BG)
    canvas.rect(0, letter[1] - 22, letter[0], 22, fill=1, stroke=0)
    canvas.setFillColor(HexColor("#1a1208"))
    canvas.setFont(SANS_B, 7)
    canvas.drawCentredString(
        letter[0] / 2.0,
        letter[1] - 14,
        "RESEARCH ONLY — NOT A MEDICAL DEVICE — NOT A PROTOCOL",
    )
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.4)
    canvas.line(0.75 * inch, 0.55 * inch, letter[0] - 0.75 * inch, 0.55 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont(SANS, 8)
    canvas.drawString(0.75 * inch, 0.38 * inch, "Ogbonna — CONFLUENCE thesis findings")
    canvas.drawRightString(
        letter[0] - 0.75 * inch,
        0.38 * inch,
        "research only  ·  %d" % doc.page,
    )
    canvas.restoreState()


def build() -> Path:
    register_fonts()
    s = styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.75 * inch,
        title=TITLE,
        author=AUTHOR,
        subject="Research technical report — not a medical device",
        creator="Project Confluence scripts/build_thesis_pdf.py",
        keywords="computational oncology, evidence gates, OnCo, CONFLUENCE",
    )

    story = []
    # Scholar wants the paper title as the largest text on page one (header bar is 7pt).
    story.append(Paragraph(html.escape(TITLE), s["title"]))
    story.append(Paragraph(html.escape(AUTHOR), s["author"]))
    story.append(Paragraph(DATE_DISPLAY, s["meta"]))
    story.append(Paragraph(html.escape(INSTITUTION), s["meta"]))
    story.append(
        Paragraph(
            'HTML: <link href="%s">%s</link> · PDF: <link href="%s">%s</link>'
            % (HTML_URL, HTML_URL, PDF_URL, PDF_URL),
            s["meta"],
        )
    )
    story.append(Spacer(1, 10))
    story.append(disclaimer_table(s))
    story.append(Spacer(1, 14))
    story.append(Paragraph("Abstract", s["h1"]))
    story.append(Paragraph(md_inline(ABSTRACT), s["body"]))

    raw = MANUSCRIPT.read_text(encoding="utf-8")
    blocks = parse_manuscript(raw)
    skip_next_abstract = True
    for kind, text in blocks:
        if kind == "subtitle":
            if text == TITLE:
                continue
            story.append(Paragraph(md_inline(text), s["meta"]))
            continue
        if kind == "meta":
            story.append(Paragraph(md_inline(text), s["meta"]))
            continue
        if kind == "h1":
            heading = text
            if heading.upper().startswith("ABSTRACT") and skip_next_abstract:
                skip_next_abstract = False
                continue
            story.append(Paragraph(md_inline(heading), s["h1"]))
            continue
        if kind == "h2":
            story.append(Paragraph(md_inline(text), s["h2"]))
            continue
        if kind == "ref":
            story.append(Paragraph(md_inline(text), s["ref"]))
            continue
        if kind == "p":
            lowered = text.lower()
            if "word export with superscript" in lowered:
                continue
            story.append(Paragraph(md_inline(text), s["body"]))

    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Source manuscript: docs/manuscript/ONCO_CONFLUENCE_THESIS_FINDINGS.md. "
            "Public HTML: %s. Attribution: Data from OnCo (onco.cc), CC BY-NC 4.0."
            % HTML_URL,
            s["meta"],
        )
    )

    doc.build(story, onFirstPage=add_page_bits, onLaterPages=add_page_bits)
    size = OUTPUT.stat().st_size
    if size > 5 * 1024 * 1024:
        raise SystemExit("PDF exceeds Google Scholar's 5MB file-size guideline")
    print("wrote %s (%d bytes)" % (OUTPUT, size))
    return OUTPUT


if __name__ == "__main__":
    build()

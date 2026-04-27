"""Build .docx and .pdf artifacts from a translated body of text.

Per docs/TRANSLATE_CONTRACT.md the output documents are deliberately
plain — no markdown rendering, just paragraphs split on blank lines,
prefixed with a small title block (original filename + target language +
"Translated by Praxis").

RTL handling: when the target language is in `RTL_LANGUAGE_CODES`, the
.docx body paragraphs get `paragraph_format.rtl = True`. reportlab does
not have a one-line RTL primitive in the version we ship; we right-align
the body for RTL targets and document the V1 limitation in the contract.
"""

from __future__ import annotations

import io

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.text.paragraph import Paragraph as DocxParagraph
from lxml import etree
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# ISO-639 codes whose scripts run right-to-left. Kept narrow on purpose —
# the contract only requires Arabic for V1; Hebrew is a likely near-term
# addition so it lives here too.
RTL_LANGUAGE_CODES: frozenset[str] = frozenset({"ar", "he"})

_FOOTER_LINE = "Translated by Praxis"


def _is_rtl(target_language_code: str) -> bool:
    return target_language_code.lower() in RTL_LANGUAGE_CODES


def _mark_paragraph_rtl(paragraph: DocxParagraph) -> None:
    """Set `<w:bidi/>` on the paragraph's pPr so Word renders it right-to-left.

    python-docx 1.x doesn't expose RTL on `ParagraphFormat`, so we have to
    drop into OOXML and add the bidi element directly. Ordering matters in
    the OOXML schema — bidi must precede `<w:rPr>` if present — but Word
    is lenient with sibling order in practice. Idempotent: skips if a
    bidi element is already present.
    """
    pPr = paragraph._p.get_or_add_pPr()  # noqa: SLF001 — required private API
    if pPr.find(qn("w:bidi")) is not None:
        return
    bidi = etree.SubElement(pPr, qn("w:bidi"))
    bidi.set(qn("w:val"), "1")


def _split_paragraphs(body: str) -> list[str]:
    """Split body text on blank lines, preserving order and dropping empties."""
    if not body:
        return []
    norm = body.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs: list[str] = []
    current: list[str] = []
    for line in norm.split("\n"):
        if line.strip() == "":
            if current:
                paragraphs.append("\n".join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append("\n".join(current).strip())
    return [p for p in paragraphs if p]


def build_docx(title: str, body: str, target_language_code: str) -> bytes:
    """Render a .docx with a header block and body paragraphs.

    The header carries the original filename (`title`), target language
    label, and a "Translated by Praxis" footer line. RTL targets get
    paragraph-level `rtl = True` and right alignment.
    """
    rtl = _is_rtl(target_language_code)
    doc = Document()

    # Title block — heading style, then a small subtitle line.
    heading = doc.add_heading(title, level=1)
    if rtl:
        _mark_paragraph_rtl(heading)
        heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    subtitle = doc.add_paragraph()
    subtitle_run = subtitle.add_run(_FOOTER_LINE)
    subtitle_run.italic = True
    subtitle_run.font.size = Pt(10)
    if rtl:
        _mark_paragraph_rtl(subtitle)
        subtitle.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    # Empty separator paragraph keeps the body away from the title block.
    doc.add_paragraph("")

    for para_text in _split_paragraphs(body):
        para = doc.add_paragraph(para_text)
        para.style = doc.styles["Normal"]
        for run in para.runs:
            run.font.size = Pt(11)
        if rtl:
            _mark_paragraph_rtl(para)
            para.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def build_pdf(title: str, body: str, target_language_code: str) -> bytes:
    """Render a .pdf using reportlab's platypus flowables.

    Letter page, 1" margins, Helvetica 11pt body, Helvetica-Bold 14pt for
    the heading. RTL languages get a right-aligned body — full bidi
    shaping is out of scope for V1 (see contract's known limitation).
    """
    rtl = _is_rtl(target_language_code)
    align = TA_RIGHT if rtl else TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
        title=title,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TranslateTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        alignment=align,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "TranslateSubtitle",
        parent=styles["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=10,
        leading=12,
        alignment=align,
        spaceAfter=14,
    )
    body_style = ParagraphStyle(
        "TranslateBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        alignment=align,
        spaceAfter=8,
    )

    flowables: list = [
        Paragraph(_pdf_escape(title), title_style),
        Paragraph(_pdf_escape(_FOOTER_LINE), subtitle_style),
        Spacer(1, 6),
    ]
    for para_text in _split_paragraphs(body):
        flowables.append(Paragraph(_pdf_escape(para_text), body_style))

    doc.build(flowables)
    return buf.getvalue()


def _pdf_escape(text: str) -> str:
    """Escape characters that reportlab's mini-HTML parser interprets.

    Paragraph() treats `<`, `>`, and `&` as markup; convert them so a body
    line containing e.g. `<patient>` doesn't crash the build with a
    paraparser exception.
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

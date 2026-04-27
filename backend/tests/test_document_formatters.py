import io

import pytest
from docx import Document
from docx.oxml.ns import qn

from anna_chat.document_formatters import (
    RTL_LANGUAGE_CODES,
    _split_paragraphs,
    build_docx,
    build_pdf,
)


def _has_bidi(paragraph) -> bool:
    """True if the paragraph has a `<w:bidi/>` element in its pPr (RTL marker)."""
    pPr = paragraph._p.find(qn("w:pPr"))
    if pPr is None:
        return False
    return pPr.find(qn("w:bidi")) is not None


def test_split_paragraphs_blank_line_boundaries():
    body = "First para line one.\nFirst para line two.\n\nSecond para.\n\n\nThird para."
    paras = _split_paragraphs(body)
    assert len(paras) == 3
    assert paras[0].startswith("First para line one")
    assert paras[1] == "Second para."
    assert paras[2] == "Third para."


def test_split_paragraphs_empty_input_returns_empty_list():
    assert _split_paragraphs("") == []
    assert _split_paragraphs("   \n\n   ") == []


def test_build_docx_returns_valid_docx_with_title_and_body():
    title = "Crank et al 2021.pdf (Spanish)"
    body = "Hola.\n\nEsto es un párrafo.\n\nÚltimo párrafo."
    raw = build_docx(title, body, target_language_code="es")
    assert isinstance(raw, bytes)
    assert len(raw) > 0
    # Re-open with python-docx to confirm it's a valid .docx and the title
    # + body landed in the document.
    doc = Document(io.BytesIO(raw))
    rendered = "\n".join(p.text for p in doc.paragraphs)
    assert title in rendered
    assert "Translated by Praxis" in rendered
    assert "Hola." in rendered
    assert "Último párrafo." in rendered


def test_build_docx_rtl_target_sets_rtl_paragraph_format():
    raw = build_docx(
        "Doc title (Arabic)",
        "اختبار.\n\nجملة ثانية.",
        target_language_code="ar",
    )
    doc = Document(io.BytesIO(raw))
    # At least one paragraph should carry `<w:bidi/>` when the target is
    # Arabic. python-docx 1.x doesn't expose RTL on the public API, so we
    # check the OOXML directly.
    assert any(_has_bidi(p) for p in doc.paragraphs), (
        "expected at least one paragraph with <w:bidi/> set for ar target"
    )


def test_build_docx_non_rtl_target_does_not_set_rtl():
    raw = build_docx(
        "Doc title (Spanish)",
        "Hola.\n\nMundo.",
        target_language_code="es",
    )
    doc = Document(io.BytesIO(raw))
    # No paragraph should be flagged RTL when the target is LTR.
    assert not any(_has_bidi(p) for p in doc.paragraphs)


def test_build_pdf_returns_valid_pdf_bytes():
    title = "Crank et al 2021.pdf (Spanish)"
    body = "Hola.\n\nEsto es un párrafo.\n\nÚltimo párrafo."
    raw = build_pdf(title, body, target_language_code="es")
    assert isinstance(raw, bytes)
    assert len(raw) > 0
    # PDF header — every valid PDF starts with `%PDF-`.
    assert raw[:5] == b"%PDF-"
    # `%%EOF` should appear near the end of any well-formed PDF.
    assert b"%%EOF" in raw[-1024:]


def test_build_pdf_handles_empty_body():
    raw = build_pdf("Empty (Spanish)", "", target_language_code="es")
    assert raw[:5] == b"%PDF-"


def test_build_pdf_escapes_html_like_tokens_in_body():
    """If the source contains `<patient>`, the reportlab paraparser must
    not treat it as a tag — `_pdf_escape` should keep `build_pdf` from
    raising."""
    body = "Notes: <patient> & <provider> discussed > 3 options."
    raw = build_pdf("Notes (Spanish)", body, target_language_code="es")
    assert raw[:5] == b"%PDF-"


def test_arabic_in_rtl_set():
    """Sanity check: contract requires Arabic to be RTL."""
    assert "ar" in RTL_LANGUAGE_CODES


@pytest.mark.parametrize("lang", ["es", "zh", "vi", "fr", "ja", "en"])
def test_build_docx_supports_multiple_target_languages(lang: str):
    raw = build_docx(
        f"Doc ({lang})",
        "Body content for the document.",
        target_language_code=lang,
    )
    doc = Document(io.BytesIO(raw))
    rendered = "\n".join(p.text for p in doc.paragraphs)
    assert "Body content for the document." in rendered

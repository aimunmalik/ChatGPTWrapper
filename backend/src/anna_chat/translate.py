"""Bedrock-backed document translation.

Per docs/TRANSLATE_CONTRACT.md, the source text is split into paragraph-aware
windows of ≤3000 input tokens with 200-token overlap, each window is
translated by a single Bedrock call, and the per-chunk outputs are
concatenated with `\\n\\n` between them. Overlap regions get a best-effort
de-duplication on line prefixes — the contract is explicit that perfect
dedup isn't worth the complexity.

Nothing in this module logs translated text, source text, or raw chunks —
counts and IDs only, per the existing JsonFormatter PHI rules.
"""

from __future__ import annotations

from dataclasses import dataclass

from anna_chat.bedrock_client import BedrockClient
from anna_chat.chunking import chunk_text

TARGET_TOKENS = 3000
OVERLAP_TOKENS = 200
MAX_TOKENS_PER_CALL = 4096

SYSTEM_PROMPT_TEMPLATE = (
    "You are a professional medical translator. Translate the user's text to "
    "{target_language_label}.\n\n"
    "The source was extracted from a PDF or Word document and may have lost "
    "its original formatting. Reconstruct structure in the OUTPUT using "
    "lightweight Markdown so the final document is readable:\n"
    "- Lines that are clearly section headings or titles → prefix with `# ` "
    "(major heading) or `## ` (subsection).\n"
    "- Bulleted lists → prefix each item with `- `.\n"
    "- Numbered lists → prefix items with `1. `, `2. `, etc.\n"
    "- Use `**bold**` for emphasis only where the source clearly emphasizes "
    "(e.g. labels, defined terms).\n"
    "- Use `*italic*` for citations, paper titles, or scientific names.\n"
    "- Otherwise output paragraphs separated by a blank line.\n\n"
    "Be conservative with structural inference — when in doubt, output a "
    "regular paragraph. Maintain person-first language. Translate inline; "
    "do NOT add introductions, commentary, summaries, or notes about the "
    "translation. Output ONLY the translation in Markdown."
)


@dataclass(frozen=True)
class TranslationResult:
    text: str
    input_tokens: int
    output_tokens: int


def _dedupe_overlap(prev_tail: str, next_text: str) -> str:
    """Strip the leading line of `next_text` if it duplicates the tail of `prev_tail`.

    The chunker carries the last `overlap_tokens` worth of words into the next
    window so cross-boundary context survives. After translation, the model
    typically renders that overlap as the start of its output, which then
    duplicates whatever the previous chunk already produced. We pop at most one
    leading line that re-appears verbatim in the previous chunk's tail —
    cheap, safe, and good enough per the contract.
    """
    if not prev_tail or not next_text:
        return next_text
    next_lines = next_text.splitlines()
    if not next_lines:
        return next_text
    first_line = next_lines[0].strip()
    if first_line and first_line in prev_tail:
        return "\n".join(next_lines[1:])
    return next_text


def translate_text(
    text: str,
    target_language_label: str,
    *,
    bedrock: BedrockClient,
) -> TranslationResult:
    """Translate `text` to the target language, chunking through Bedrock.

    Returns the concatenated translation plus aggregate token usage. Empty
    or whitespace-only input returns an empty result with zero tokens.
    """
    windows = chunk_text(
        text,
        target_tokens=TARGET_TOKENS,
        overlap_tokens=OVERLAP_TOKENS,
    )
    if not windows:
        return TranslationResult(text="", input_tokens=0, output_tokens=0)

    system = SYSTEM_PROMPT_TEMPLATE.format(
        target_language_label=target_language_label
    )

    pieces: list[str] = []
    input_tokens_total = 0
    output_tokens_total = 0
    prev_tail = ""

    for window in windows:
        resp = bedrock.invoke(
            messages=[{"role": "user", "content": window.text}],
            system=system,
            max_tokens=MAX_TOKENS_PER_CALL,
            temperature=0.2,
        )
        chunk_out = resp.text
        if pieces:
            chunk_out = _dedupe_overlap(prev_tail, chunk_out)
        pieces.append(chunk_out)
        # Keep at most the last 1500 chars as the dedupe window — enough to
        # catch a duplicated opening line without scanning the whole string.
        prev_tail = chunk_out[-1500:]
        input_tokens_total += int(resp.input_tokens or 0)
        output_tokens_total += int(resp.output_tokens or 0)

    joined = "\n\n".join(p.strip() for p in pieces if p and p.strip())
    return TranslationResult(
        text=joined,
        input_tokens=input_tokens_total,
        output_tokens=output_tokens_total,
    )

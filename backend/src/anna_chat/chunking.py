"""Paragraph-aware sliding-window chunker.

Pure text input to `chunk_text()` — the caller (the ingest handler) knows the
page / section metadata and attaches it after the fact if available.

Strategy:
  1. Split input on blank lines into paragraphs.
  2. Pack paragraphs into windows capped at `target_tokens` (approx).
  3. When closing a window, carry the last `overlap_tokens` worth of text into
     the next window so cross-paragraph context survives boundaries.

Token count is a rough heuristic: `len(text.split()) * 1.3`. Good enough for
bounding decisions. Titan v2 is billed on server-side token count — this
function doesn't influence billing, only chunking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN_PER_WORD = 1.3
_PARA_SPLIT = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class ChunkWindow:
    text: str
    approx_tokens: int
    page_number: int | None = None
    section_title: str | None = None


def approx_token_count(text: str) -> int:
    """Heuristic token count: word count × 1.3 (empty → 0)."""
    if not text:
        return 0
    return int(len(text.split()) * TOKEN_PER_WORD)


def _split_paragraphs(text: str) -> list[str]:
    # Normalize line endings and trim. Empty paragraphs are dropped.
    norm = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not norm:
        return []
    return [p.strip() for p in _PARA_SPLIT.split(norm) if p.strip()]


def _carry_overlap(window_text: str, overlap_tokens: int) -> str:
    """Return the tail of `window_text` worth roughly `overlap_tokens` tokens.

    We walk words from the end until the approximate token budget is reached.
    """
    if overlap_tokens <= 0 or not window_text:
        return ""
    words = window_text.split()
    if not words:
        return ""
    target_words = max(int(overlap_tokens / TOKEN_PER_WORD), 1)
    tail = words[-target_words:]
    return " ".join(tail)


def chunk_text(
    text: str,
    *,
    target_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[ChunkWindow]:
    """Split `text` into overlapping windows of approximately `target_tokens`.

    - Returns [] for empty / whitespace-only input.
    - If the whole document fits in one window, returns a single window.
    - A single paragraph larger than `target_tokens` is force-split on word
      boundaries into windows up to `target_tokens` each, preserving overlap.
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    windows: list[ChunkWindow] = []
    current_parts: list[str] = []
    current_tokens = 0
    carry = ""

    def _flush() -> None:
        nonlocal current_parts, current_tokens, carry
        if not current_parts:
            return
        body = "\n\n".join(current_parts).strip()
        if not body:
            current_parts = []
            current_tokens = 0
            return
        windows.append(
            ChunkWindow(text=body, approx_tokens=approx_token_count(body))
        )
        carry = _carry_overlap(body, overlap_tokens)
        current_parts = []
        current_tokens = 0

    for para in paragraphs:
        para_tokens = approx_token_count(para)

        # Oversized paragraph — split on words.
        if para_tokens > target_tokens:
            # Flush any in-progress window first.
            _flush()
            words = para.split()
            words_per_window = max(int(target_tokens / TOKEN_PER_WORD), 1)
            overlap_words = max(int(overlap_tokens / TOKEN_PER_WORD), 0)
            i = 0
            prev_tail = carry
            while i < len(words):
                slice_words = words[i : i + words_per_window]
                body_parts = []
                if prev_tail:
                    body_parts.append(prev_tail)
                body_parts.append(" ".join(slice_words))
                body = " ".join(body_parts).strip()
                windows.append(
                    ChunkWindow(text=body, approx_tokens=approx_token_count(body))
                )
                prev_tail = (
                    " ".join(slice_words[-overlap_words:]) if overlap_words else ""
                )
                i += words_per_window
            carry = prev_tail
            continue

        # Starting a fresh window: seed with overlap carried from the last one.
        if not current_parts and carry:
            current_parts.append(carry)
            current_tokens += approx_token_count(carry)
            carry = ""

        # If adding this paragraph would exceed target, flush first.
        if current_parts and current_tokens + para_tokens > target_tokens:
            _flush()
            if carry:
                current_parts.append(carry)
                current_tokens += approx_token_count(carry)
                carry = ""

        current_parts.append(para)
        current_tokens += para_tokens

    _flush()
    return windows

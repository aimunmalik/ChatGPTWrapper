"""Knowledge base retrieval — synchronous cosine-similarity top-K lookup.

Called from the chat handler before Bedrock is invoked. All chunks live in
memory for the lifetime of the Lambda container; we re-scan DDB every 5
minutes (or on cache miss) to pick up new or deleted docs without paying the
DDB cost per chat turn.

No numpy — 1024-dim × ~1K chunks in pure Python is ~50 ms, well under budget,
and keeps the Lambda zip small. Vectors are already unit-normalized by Titan
Embed v2 (we pass `normalize: true`), so cosine similarity is just dot product.

PHI note: this module MUST NOT log `query` or `chunk.chunkText` or embeddings.
Log only scalar metadata (top_score, chunks_returned, latency_ms).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from anna_chat.embeddings import EmbeddingsClient
from anna_chat.kb_repo import Chunk, KbRepo
from anna_chat.logging_config import get_logger

logger = get_logger(__name__)

CACHE_TTL_SECONDS = 5 * 60

# Module-level cache shared across warm-invocation handler calls.
_CACHED_CHUNKS: list[Chunk] | None = None
_CACHED_AT: float = 0.0


@dataclass(frozen=True)
class RetrievedChunk:
    kb_doc_id: str
    doc_title: str
    source_type: str
    chunk_idx: int
    chunk_text: str
    page_number: int | None
    section_title: str | None
    score: float


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length float vectors.

    We don't assume the vectors are pre-normalized because callers may pass
    vectors read from DDB (where Decimal → float coercion introduces tiny
    drift) or synthetic vectors in tests.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(n):
        va = a[i]
        vb = b[i]
        dot += va * vb
        norm_a += va * va
        norm_b += vb * vb
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _load_chunks(repo: KbRepo, *, now: float | None = None) -> list[Chunk]:
    """Return cached chunks, refreshing from DDB if the cache is stale."""
    global _CACHED_CHUNKS, _CACHED_AT
    now = now if now is not None else time.monotonic()
    if _CACHED_CHUNKS is not None and (now - _CACHED_AT) < CACHE_TTL_SECONDS:
        return _CACHED_CHUNKS
    chunks = repo.scan_all_chunks()
    _CACHED_CHUNKS = chunks
    _CACHED_AT = now
    return chunks


def invalidate_cache() -> None:
    """Drop the in-memory chunk cache (used by tests and by delete_doc)."""
    global _CACHED_CHUNKS, _CACHED_AT
    _CACHED_CHUNKS = None
    _CACHED_AT = 0.0


def retrieve(
    query: str,
    *,
    embeddings: EmbeddingsClient,
    repo: KbRepo,
    top_k: int = 5,
    min_score: float = 0.35,
) -> list[RetrievedChunk]:
    """Return up to `top_k` chunks whose cosine similarity ≥ `min_score`.

    Returns [] when the KB is empty or no chunks clear the threshold. The
    returned `RetrievedChunk`s have their embedding stripped — callers only
    need it for scoring, and hanging onto 1024 floats × K across the chat
    turn is wasteful.
    """
    started = time.monotonic()
    if not query or not query.strip():
        return []

    query_vec = embeddings.embed(query)
    all_chunks = _load_chunks(repo)
    if not all_chunks:
        latency_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "kb_retrieve_empty",
            extra={
                "chunksReturned": 0,
                "chunksScanned": 0,
                "topScore": 0.0,
                "latencyMs": latency_ms,
            },
        )
        return []

    scored: list[tuple[float, Chunk]] = []
    for chunk in all_chunks:
        score = _cosine(query_vec, chunk.embedding)
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)

    top: list[RetrievedChunk] = []
    for score, chunk in scored[:top_k]:
        if score < min_score:
            continue
        top.append(
            RetrievedChunk(
                kb_doc_id=chunk.kbDocId,
                doc_title=chunk.docTitle,
                source_type=chunk.sourceType,
                chunk_idx=chunk.chunkIdx,
                chunk_text=chunk.chunkText,
                page_number=chunk.pageNumber,
                section_title=chunk.sectionTitle,
                score=float(score),
            )
        )

    latency_ms = int((time.monotonic() - started) * 1000)
    top_score = top[0].score if top else (scored[0][0] if scored else 0.0)
    logger.info(
        "kb_retrieve_complete",
        extra={
            "chunksReturned": len(top),
            "chunksScanned": len(all_chunks),
            "topScore": round(float(top_score), 4),
            "latencyMs": latency_ms,
        },
    )
    return top

"""Shared chat building blocks used by BOTH the synchronous chat handler
(`handlers/chat.py`, route POST /chat) and the async streaming chat worker
(`handlers/chat_worker.py`).

Holds the system prompt, the allowed-model allowlist, KB retrieval, the
`<knowledge>` block formatter, attachment prepending, and the `build_turn`
helper that assembles everything Bedrock needs for one turn.

PHI rule: helpers here return content destined for Bedrock but NEVER log it.
Singletons are cached per-process; each Lambda (chat vs chat-worker) gets its
own cache, so sharing this module across both functions is safe.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from anna_chat import kb_retrieve
from anna_chat.attachments_repo import AttachmentsRepo
from anna_chat.ddb import Repository
from anna_chat.embeddings import EmbeddingsClient
from anna_chat.kb_repo import KbRepo
from anna_chat.kb_retrieve import RetrievedChunk
from anna_chat.logging_config import get_logger
from anna_chat.settings import Settings

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Praxis, a work assistant for staff at ANNA Health — the Allied "
    "Network for Neurodevelopmental Advancement, an Applied Behavior Analysis "
    "(ABA) provider for children with autism.\n\n"
    "Be helpful with anything a colleague would reasonably ask. That includes "
    "clinical tasks (treatment plans, session notes, BIPs, parent "
    "communication, assessment interpretation, ABA questions) AND any other "
    "work task — translating documents, drafting HR or marketing copy, "
    "summarizing, comparing, analyzing spreadsheets, coding, research, "
    "brainstorming, math, rewriting, whatever. Do not refuse a request "
    "because it is non-clinical. Adapt to the task.\n\n"
    "When the task IS clinical: be concise and clinically accurate, use "
    "person-first language by default, flag anything that calls for a "
    "licensed professional's judgment, and never fabricate citations or "
    "guidelines.\n\n"
    "When the task is NOT clinical: apply the same standards you would in "
    "any careful work — accuracy over speed, flag uncertainty, do not make "
    "up facts — but do not impose clinical framing where it does not belong.\n\n"
    "If the user attaches documents, treat them as primary context and "
    "reference them specifically.\n\n"
    "When the <knowledge> block contains relevant material, prefer it over "
    "your general knowledge and cite the source number in your response like "
    "[1] or [Source 2]. When no relevant material is returned, answer from "
    "general knowledge and say so briefly."
)

ALLOWED_MODELS: set[str] = {
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-opus-4-7",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def repo() -> Repository:
    s = settings()
    return Repository(
        conversations_table=s.conversations_table,
        messages_table=s.messages_table,
        region=s.aws_region,
        message_ttl_days=s.message_ttl_days,
    )


@lru_cache(maxsize=1)
def attachments_repo() -> AttachmentsRepo | None:
    s = settings()
    if not s.attachments_table:
        return None
    return AttachmentsRepo(
        attachments_table=s.attachments_table,
        region=s.aws_region,
        message_ttl_days=s.message_ttl_days,
    )


@lru_cache(maxsize=1)
def kb_repo() -> KbRepo | None:
    s = settings()
    if not s.kb_table:
        return None
    return KbRepo(kb_table=s.kb_table, region=s.aws_region)


@lru_cache(maxsize=1)
def embeddings() -> EmbeddingsClient | None:
    s = settings()
    if not s.kb_table:
        return None
    return EmbeddingsClient(region=s.aws_region)


def format_knowledge_block(retrieved: list[RetrievedChunk]) -> str:
    """Render the <knowledge>...</knowledge> block per the KB contract."""
    if not retrieved:
        return (
            "<knowledge>\n"
            "No relevant material found in the ANNA knowledge base.\n"
            "</knowledge>"
        )
    parts: list[str] = []
    for i, chunk in enumerate(retrieved, start=1):
        header = f"[Source {i}] {chunk.doc_title}"
        extras: list[str] = []
        if chunk.section_title:
            extras.append(f"section: {chunk.section_title}")
        if chunk.page_number is not None:
            extras.append(f"page {chunk.page_number}")
        if extras:
            header += " — " + ", ".join(extras)
        parts.append(f"{header}\n{chunk.chunk_text}")
    body = "\n\n---\n\n".join(parts)
    return f"<knowledge>\n{body}\n</knowledge>"


def retrieve_sources(user_message: str) -> list[RetrievedChunk]:
    """Run KB retrieval if configured; swallow errors to keep chat alive."""
    kb = kb_repo()
    emb = embeddings()
    if kb is None or emb is None:
        return []
    try:
        return kb_retrieve.retrieve(user_message, embeddings=emb, repo=kb)
    except Exception as exc:
        # Retrieval is best-effort — if Bedrock embeddings aren't enabled or
        # the KB table is cold, we still want the chat turn to succeed.
        logger.error("kb_retrieve_failed", extra={"errorType": type(exc).__name__})
        return []


def sources_payload(retrieved: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Shape retrieved chunks into the wire `sources` array (citation order)."""
    return [
        {
            "index": i + 1,
            "kbDocId": c.kb_doc_id,
            "docTitle": c.doc_title,
            "sourceType": c.source_type,
            "pageNumber": c.page_number,
            "score": round(c.score, 3),
        }
        for i, c in enumerate(retrieved)
    ]


def prepend_attachments(
    conversation_id: str, user_message: str
) -> tuple[str, list[dict[str, Any]]]:
    """Return (possibly-augmented message, per-attachment log metadata).

    The returned log metadata contains only ids and byte sizes — never content.
    """
    repo_ = attachments_repo()
    if repo_ is None:
        return user_message, []
    atts = repo_.list_for_conversation(conversation_id=conversation_id, status="ready")
    if not atts:
        return user_message, []
    blocks: list[str] = []
    meta: list[dict[str, Any]] = []
    for att in atts:
        text = att.extractedText or ""
        blocks.append(
            f'<attachment filename="{att.filename}" contentType="{att.contentType}">\n'
            f"{text}\n"
            f"</attachment>"
        )
        meta.append(
            {
                "attachmentId": att.attachmentId,
                "sizeBytes": att.sizeBytes,
                "extractedBytes": len(text.encode("utf-8")),
            }
        )
    combined = "\n\n".join(blocks) + "\n\n" + user_message
    return combined, meta


def build_turn(
    conversation_id: str, user_message: str
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Assemble one chat turn for Bedrock.

    Returns:
      augmented_for_bedrock — the user message with the <knowledge> block and
                              any <attachment> blocks prepended (what we SEND
                              to Bedrock; never what we persist).
      sources               — the wire `sources` array for the response.
      attachment_meta       — PHI-safe per-attachment log metadata.

    RAG retrieval runs on the RAW user message (before attachment augmentation)
    so the semantic search matches the question, not attachment content.
    """
    retrieved = retrieve_sources(user_message)
    knowledge_block = format_knowledge_block(retrieved)
    sources = sources_payload(retrieved)
    augmented_message, attachment_meta = prepend_attachments(conversation_id, user_message)
    augmented_for_bedrock = knowledge_block + "\n\n" + augmented_message
    return augmented_for_bedrock, sources, attachment_meta

import time
from functools import lru_cache
from typing import Any

from anna_chat import kb_retrieve
from anna_chat.attachments_repo import AttachmentsRepo
from anna_chat.bedrock_client import BedrockClient
from anna_chat.ddb import Repository
from anna_chat.embeddings import EmbeddingsClient
from anna_chat.http import HttpError, authenticate, error, ok, parse_json_body
from anna_chat.kb_repo import KbRepo
from anna_chat.kb_retrieve import RetrievedChunk
from anna_chat.logging_config import configure_logging, get_logger
from anna_chat.settings import Settings

configure_logging()
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
def _settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def _bedrock() -> BedrockClient:
    s = _settings()
    return BedrockClient(region=s.aws_region, model_id=s.bedrock_model_id)


@lru_cache(maxsize=1)
def _repo() -> Repository:
    s = _settings()
    return Repository(
        conversations_table=s.conversations_table,
        messages_table=s.messages_table,
        region=s.aws_region,
        message_ttl_days=s.message_ttl_days,
    )


@lru_cache(maxsize=1)
def _attachments_repo() -> AttachmentsRepo | None:
    s = _settings()
    if not s.attachments_table:
        return None
    return AttachmentsRepo(
        attachments_table=s.attachments_table,
        region=s.aws_region,
        message_ttl_days=s.message_ttl_days,
    )


@lru_cache(maxsize=1)
def _kb_repo() -> KbRepo | None:
    s = _settings()
    if not s.kb_table:
        return None
    return KbRepo(kb_table=s.kb_table, region=s.aws_region)


@lru_cache(maxsize=1)
def _embeddings() -> EmbeddingsClient | None:
    s = _settings()
    if not s.kb_table:
        return None
    return EmbeddingsClient(region=s.aws_region)


def _format_knowledge_block(retrieved: list[RetrievedChunk]) -> str:
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


def _retrieve_sources(user_message: str) -> list[RetrievedChunk]:
    """Run KB retrieval if configured; swallow errors to keep chat alive."""
    kb_repo = _kb_repo()
    embeddings = _embeddings()
    if kb_repo is None or embeddings is None:
        return []
    try:
        return kb_retrieve.retrieve(
            user_message, embeddings=embeddings, repo=kb_repo
        )
    except Exception as exc:
        # Retrieval is best-effort — if Bedrock embeddings aren't enabled or
        # the KB table is cold, we still want the user's chat turn to succeed.
        logger.error(
            "kb_retrieve_failed",
            extra={"errorType": type(exc).__name__},
        )
        return []


def _prepend_attachments(
    attachments_repo: AttachmentsRepo | None, conversation_id: str, user_message: str
) -> tuple[str, list[dict[str, Any]]]:
    """Return (possibly-augmented message, per-attachment log metadata).

    The returned log metadata contains only ids and byte sizes — never content.
    """
    if attachments_repo is None:
        return user_message, []
    atts = attachments_repo.list_for_conversation(
        conversation_id=conversation_id, status="ready"
    )
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


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    start = time.monotonic()
    try:
        user = authenticate(event, _settings())
        body = parse_json_body(event)

        user_message = (body.get("message") or "").strip()
        if not user_message:
            raise HttpError(400, "message is required")
        if len(user_message) > 20000:
            raise HttpError(400, "message too long")

        requested_model = body.get("model")
        if requested_model is not None and requested_model not in ALLOWED_MODELS:
            raise HttpError(400, f"unsupported model: {requested_model}")
        model_id = requested_model or _settings().bedrock_model_id

        repo = _repo()
        bedrock = _bedrock()

        conversation_id = body.get("conversationId")
        if conversation_id:
            conv = repo.get_conversation(user_id=user.sub, conversation_id=conversation_id)
            if not conv:
                raise HttpError(404, "conversation not found")
        else:
            title = user_message[:80]
            conv = repo.create_conversation(
                user_id=user.sub, title=title, model=model_id
            )

        history = repo.recent_turns_for_model(conversation_id=conv.conversationId)

        # RAG retrieval — run on the raw user message BEFORE it's augmented
        # with attachment blocks. We want the semantic search to match the
        # user's question, not attachment content.
        retrieved = _retrieve_sources(user_message)
        knowledge_block = _format_knowledge_block(retrieved)
        sources = [
            {
                "index": i + 1,
                "docTitle": c.doc_title,
                "sourceType": c.source_type,
                "pageNumber": c.page_number,
                "score": round(c.score, 3),
            }
            for i, c in enumerate(retrieved)
        ]

        # Persist the raw user message (what the user actually typed), but send
        # Bedrock a version with <knowledge> + <attachment> blocks prepended so
        # it can reason over retrieved and user-attached context. Storing the
        # augmented version in DDB would leak KB content into the conversation
        # history and surface it in the UI — we keep the stored message clean
        # and rebuild the augmented view each turn.
        augmented_message, attachment_meta = _prepend_attachments(
            _attachments_repo(), conv.conversationId, user_message
        )
        # Prepend the knowledge block ahead of attachments and the raw message.
        augmented_for_bedrock = knowledge_block + "\n\n" + augmented_message
        history.append({"role": "user", "content": augmented_for_bedrock})

        repo.append_message(
            conversation_id=conv.conversationId,
            user_id=user.sub,
            role="user",
            content=user_message,
            model=model_id,
        )

        bedrock_resp = bedrock.invoke(
            messages=history, system=SYSTEM_PROMPT, model_id=model_id
        )

        assistant_msg = repo.append_message(
            conversation_id=conv.conversationId,
            user_id=user.sub,
            role="assistant",
            content=bedrock_resp.text,
            input_tokens=bedrock_resp.input_tokens,
            output_tokens=bedrock_resp.output_tokens,
            model=model_id,
            sources=sources,
        )
        repo.touch_conversation(user_id=user.sub, conversation_id=conv.conversationId)

        latency_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "chat_turn_complete",
            extra={
                "userId": user.sub,
                "conversationId": conv.conversationId,
                "messageId": assistant_msg.messageId,
                "inputTokens": bedrock_resp.input_tokens,
                "outputTokens": bedrock_resp.output_tokens,
                "model": model_id,
                "latencyMs": latency_ms,
                "stopReason": bedrock_resp.stop_reason,
                "attachments": attachment_meta,
                "attachmentCount": len(attachment_meta),
                "kbSourceCount": len(sources),
                "kbTopScore": sources[0]["score"] if sources else 0.0,
            },
        )

        return ok(
            {
                "conversationId": conv.conversationId,
                "messageId": assistant_msg.messageId,
                "assistantMessage": bedrock_resp.text,
                "tokens": {
                    "input": bedrock_resp.input_tokens,
                    "output": bedrock_resp.output_tokens,
                },
                "model": model_id,
                "sources": sources,
            }
        )
    except HttpError as exc:
        logger.info(
            "chat_http_error",
            extra={"status": exc.status, "reason": exc.message},
        )
        return error(exc.status, exc.message)
    except Exception as exc:
        logger.error(
            "chat_unhandled_error",
            extra={"errorType": type(exc).__name__},
        )
        return error(500, "internal error")

import time
from functools import lru_cache
from typing import Any

from anna_chat.attachments_repo import AttachmentsRepo
from anna_chat.bedrock_client import BedrockClient
from anna_chat.ddb import Repository
from anna_chat.http import HttpError, authenticate, error, ok, parse_json_body
from anna_chat.logging_config import configure_logging, get_logger
from anna_chat.settings import Settings

configure_logging()
logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Praxis, ANNA Health's clinical assistant. ANNA (Allied Network for "
    "Neurodevelopmental Advancement) provides Applied Behavior Analysis (ABA) services "
    "for children with autism. Help clinicians with treatment planning, note drafting, "
    "parent communication, and evidence-based practice questions. Be concise and "
    "clinically accurate. When a question is outside your scope or calls for licensed "
    "judgment, say so. Use person-first language unless the clinician requests "
    "identity-first. Do not fabricate citations."
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

        # Persist the raw user message (what the user actually typed), but send
        # Bedrock a version with <attachment> blocks prepended so it can reason
        # over extracted text. Never log the augmented content.
        augmented_message, attachment_meta = _prepend_attachments(
            _attachments_repo(), conv.conversationId, user_message
        )
        history.append({"role": "user", "content": augmented_message})

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
            }
        )
    except HttpError as exc:
        logger.info(
            "chat_http_error",
            extra={"status": exc.status, "reason": exc.message},
        )
        return error(exc.status, exc.message)
    except Exception:
        logger.exception("chat_unhandled_error")
        return error(500, "internal error")

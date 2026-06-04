r"""Async worker Lambda for streaming chat.

Invoked by `handlers/chat._start_stream` with `InvocationType=Event`. Picks up
the empty assistant message the kickoff created, streams a Bedrock completion
into it, and closes it out:

    streaming -> (periodic content updates) -> complete
                                            \-> error

The kickoff already persisted the raw user message and created the assistant
placeholder, so this worker only needs to: rebuild the model-facing history,
run KB retrieval, stream tokens, and flush the accumulating text to the message
row roughly once per second (the frontend polls and renders the partial text).

Why this exists: API Gateway HTTP APIs cap a synchronous response at 30s. A
4-part clinical question on Opus blows past that and 503s. This worker has the
Lambda's full timeout budget (15 min), so long answers finish; the browser
polls GET /chat/stream instead of holding one long request open.

PHI rules: never log message content, chunk text, or KB material — ids, counts,
token totals, and error *types* only.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from anna_chat.bedrock_client import STREAM_READ_TIMEOUT, BedrockClient
from anna_chat.handlers import chat_core
from anna_chat.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# Flush the growing message body to DynamoDB at most this often. Bounds DDB
# write volume on long answers while keeping the poll UI feeling live (~1s).
FLUSH_INTERVAL_SECONDS = 0.8


@lru_cache(maxsize=1)
def _bedrock_stream() -> BedrockClient:
    """Bedrock client tuned for streaming: a generous inter-chunk read timeout
    and NO retries (a partially-consumed stream must never be replayed)."""
    s = chat_core.settings()
    return BedrockClient(
        region=s.aws_region,
        model_id=s.bedrock_model_id,
        read_timeout=STREAM_READ_TIMEOUT,
        max_attempts=1,
    )


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    conversation_id = (event or {}).get("conversationId")
    user_id = (event or {}).get("userId")
    sort_key = (event or {}).get("sortKey")
    message_id = (event or {}).get("messageId")
    user_message = (event or {}).get("userMessage") or ""
    model_id = (event or {}).get("model") or chat_core.settings().bedrock_model_id

    if not (conversation_id and user_id and sort_key and user_message):
        logger.error(
            "chat_worker_bad_event",
            extra={
                "hasConversationId": bool(conversation_id),
                "hasUserId": bool(user_id),
                "hasSortKey": bool(sort_key),
                "hasUserMessage": bool(user_message),
            },
        )
        return {"ok": False, "reason": "missing required fields"}

    repo = chat_core.repo()
    bedrock = _bedrock_stream()
    start = time.monotonic()
    accumulated: list[str] = []

    try:
        # History already contains the raw user message (the kickoff persisted
        # it) and excludes this streaming placeholder. Swap the trailing raw
        # user turn for the augmented version (<knowledge> + <attachment>
        # blocks) so Bedrock reasons over retrieved + attached context.
        history = repo.recent_turns_for_model(conversation_id=conversation_id)
        augmented_for_bedrock, sources, attachment_meta = chat_core.build_turn(
            conversation_id, user_message
        )
        if history and history[-1]["role"] == "user":
            history[-1] = {"role": "user", "content": augmented_for_bedrock}
        else:
            history.append({"role": "user", "content": augmented_for_bedrock})

        input_tokens = 0
        output_tokens = 0
        stop_reason = "unknown"
        last_flush = time.monotonic()

        for ev in bedrock.invoke_stream(
            messages=history, system=chat_core.SYSTEM_PROMPT, model_id=model_id
        ):
            if ev["type"] == "delta":
                accumulated.append(ev["text"])
                now = time.monotonic()
                if now - last_flush >= FLUSH_INTERVAL_SECONDS:
                    repo.update_streaming_content(
                        conversation_id=conversation_id,
                        sort_key=sort_key,
                        content="".join(accumulated),
                    )
                    last_flush = now
            elif ev["type"] == "done":
                input_tokens = ev["inputTokens"]
                output_tokens = ev["outputTokens"]
                stop_reason = ev["stopReason"]

        final_text = "".join(accumulated)
        repo.finalize_message(
            conversation_id=conversation_id,
            sort_key=sort_key,
            content=final_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model_id,
            sources=sources,
            status="complete",
        )
        repo.touch_conversation(user_id=user_id, conversation_id=conversation_id)

        logger.info(
            "chat_stream_complete",
            extra={
                "userId": user_id,
                "conversationId": conversation_id,
                "messageId": message_id,
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "model": model_id,
                "stopReason": stop_reason,
                "attachmentCount": len(attachment_meta),
                "kbSourceCount": len(sources),
                "latencyMs": int((time.monotonic() - start) * 1000),
            },
        )
        return {"ok": True, "messageId": message_id}

    except Exception as exc:
        error_type = type(exc).__name__
        # Best-effort: mark the message errored so the poll terminates and the
        # UI stops spinning. Preserve whatever text streamed before the failure.
        try:
            repo.finalize_message(
                conversation_id=conversation_id,
                sort_key=sort_key,
                content="".join(accumulated),
                model=model_id,
                sources=[],
                status="error",
            )
        except Exception:
            logger.error(
                "chat_stream_finalize_failed",
                extra={
                    "userId": user_id,
                    "conversationId": conversation_id,
                    "messageId": message_id,
                },
            )
        logger.error(
            "chat_stream_failed",
            extra={
                "userId": user_id,
                "conversationId": conversation_id,
                "messageId": message_id,
                "errorType": error_type,
            },
        )
        return {"ok": False, "messageId": message_id, "errorType": error_type}

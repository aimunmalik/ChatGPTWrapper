"""HTTP handler for the chat routes.

Routes (all JWT-authenticated via the API Gateway authorizer):
  - POST /chat          synchronous, buffered turn (legacy / fallback). Bounded
                        by API Gateway's 30s integration cap — fine for short
                        turns on fast models, will 504 on long Opus answers.
  - POST /chat/stream   kickoff for a streamed turn: persists the user message,
                        creates an empty `streaming` assistant message, and
                        async-invokes the chat worker (InvocationType=Event).
                        Returns immediately with the ids needed to poll.
  - GET  /chat/stream   poll the streaming assistant message by (cid, sk) query
                        params; returns partial content + status until terminal.

The streaming pair sidesteps the 30s cap: the worker has a 15-minute budget,
so Opus and long multi-part questions complete. The browser polls instead of
holding one long request open. See docs/STREAMING_CONTRACT.md.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Any

import boto3

from anna_chat.bedrock_client import BedrockClient
from anna_chat.handlers import chat_core
from anna_chat.http import (
    AuthenticatedUser,
    HttpError,
    authenticate,
    error,
    ok,
    parse_json_body,
)
from anna_chat.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

MAX_MESSAGE_CHARS = 20000


@lru_cache(maxsize=1)
def _bedrock() -> BedrockClient:
    """Synchronous (buffered) Bedrock client for the legacy POST /chat path."""
    s = chat_core.settings()
    return BedrockClient(region=s.aws_region, model_id=s.bedrock_model_id)


@lru_cache(maxsize=1)
def _lambda():
    s = chat_core.settings()
    return boto3.client("lambda", region_name=s.aws_region)


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    try:
        user = authenticate(event, chat_core.settings())
        route_key = event.get("routeKey", "")
        if route_key == "POST /chat/stream":
            return _start_stream(event, user)
        if route_key == "GET /chat/stream":
            return _poll_stream(event, user)
        # Default: legacy synchronous chat (POST /chat).
        return _sync_chat(event, user)
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


def _validate_message(body: dict[str, Any]) -> str:
    user_message = (body.get("message") or "").strip()
    if not user_message:
        raise HttpError(400, "message is required")
    if len(user_message) > MAX_MESSAGE_CHARS:
        raise HttpError(400, "message too long")
    return user_message


def _resolve_model(body: dict[str, Any]) -> str:
    requested_model = body.get("model")
    if requested_model is not None and requested_model not in chat_core.ALLOWED_MODELS:
        raise HttpError(400, f"unsupported model: {requested_model}")
    return requested_model or chat_core.settings().bedrock_model_id


# --------------------------------------------------------------------------- #
# POST /chat — synchronous, buffered (legacy / fallback)
# --------------------------------------------------------------------------- #
def _sync_chat(event: dict[str, Any], user: AuthenticatedUser) -> dict[str, Any]:
    start = time.monotonic()
    body = parse_json_body(event)
    user_message = _validate_message(body)
    model_id = _resolve_model(body)

    repo = chat_core.repo()
    bedrock = _bedrock()

    conv = _get_or_create_conversation(repo, user, body, model_id, user_message)

    history = repo.recent_turns_for_model(conversation_id=conv.conversationId)
    augmented_for_bedrock, sources, attachment_meta = chat_core.build_turn(
        conv.conversationId, user_message
    )
    history.append({"role": "user", "content": augmented_for_bedrock})

    repo.append_message(
        conversation_id=conv.conversationId,
        user_id=user.sub,
        role="user",
        content=user_message,
        model=model_id,
    )

    bedrock_resp = bedrock.invoke(
        messages=history, system=chat_core.SYSTEM_PROMPT, model_id=model_id
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


# --------------------------------------------------------------------------- #
# POST /chat/stream — kickoff (async)
# --------------------------------------------------------------------------- #
def _start_stream(event: dict[str, Any], user: AuthenticatedUser) -> dict[str, Any]:
    body = parse_json_body(event)
    user_message = _validate_message(body)
    model_id = _resolve_model(body)

    s = chat_core.settings()
    if not s.chat_worker_function_name:
        raise HttpError(500, "chat worker not configured")

    repo = chat_core.repo()
    conv = _get_or_create_conversation(repo, user, body, model_id, user_message)

    # Persist the raw user message FIRST so it sorts before the assistant row
    # (sort key is timestamp-ordered), then create the empty assistant
    # placeholder the worker will stream into.
    repo.append_message(
        conversation_id=conv.conversationId,
        user_id=user.sub,
        role="user",
        content=user_message,
        model=model_id,
    )
    assistant = repo.create_streaming_message(
        conversation_id=conv.conversationId, user_id=user.sub, model=model_id
    )
    repo.touch_conversation(user_id=user.sub, conversation_id=conv.conversationId)

    payload = json.dumps(
        {
            "conversationId": conv.conversationId,
            "userId": user.sub,
            "sortKey": assistant.sortKey,
            "messageId": assistant.messageId,
            "userMessage": user_message,
            "model": model_id,
        }
    )
    # Event invocation returns 202 immediately; ClientError (access denied /
    # function-not-found) bubbles to the top-level handler as a 500.
    _lambda().invoke(
        FunctionName=s.chat_worker_function_name,
        InvocationType="Event",
        Payload=payload.encode("utf-8"),
    )

    logger.info(
        "chat_stream_started",
        extra={
            "userId": user.sub,
            "conversationId": conv.conversationId,
            "messageId": assistant.messageId,
            "model": model_id,
        },
    )

    return ok(
        {
            "conversationId": conv.conversationId,
            "messageId": assistant.messageId,
            "sortKey": assistant.sortKey,
            "status": "streaming",
        },
        status=202,
    )


# --------------------------------------------------------------------------- #
# GET /chat/stream — poll
# --------------------------------------------------------------------------- #
def _poll_stream(event: dict[str, Any], user: AuthenticatedUser) -> dict[str, Any]:
    qs = event.get("queryStringParameters") or {}
    conversation_id = (qs.get("cid") or "").strip()
    sort_key = (qs.get("sk") or "").strip()
    if not conversation_id or not sort_key:
        raise HttpError(400, "cid and sk query params are required")

    repo = chat_core.repo()
    # Ownership: the conversations table is keyed by (userId, conversationId),
    # so a hit proves the caller owns the conversation. Collapse miss to 404 to
    # avoid existence enumeration.
    conv = repo.get_conversation(user_id=user.sub, conversation_id=conversation_id)
    if not conv:
        raise HttpError(404, "conversation not found")

    msg = repo.get_message(conversation_id=conversation_id, sort_key=sort_key)
    if not msg or msg.userId != user.sub:
        raise HttpError(404, "message not found")

    return ok(
        {
            "conversationId": conversation_id,
            "messageId": msg.messageId,
            "sortKey": msg.sortKey,
            "status": msg.status,
            "content": msg.content,
            "sources": msg.sources,
            "tokens": {"input": msg.inputTokens, "output": msg.outputTokens},
            "model": msg.model,
        }
    )


def _get_or_create_conversation(
    repo: Any,
    user: AuthenticatedUser,
    body: dict[str, Any],
    model_id: str,
    user_message: str,
) -> Any:
    conversation_id = body.get("conversationId")
    if conversation_id:
        conv = repo.get_conversation(
            user_id=user.sub, conversation_id=conversation_id
        )
        if not conv:
            raise HttpError(404, "conversation not found")
        return conv
    return repo.create_conversation(
        user_id=user.sub, title=user_message[:80], model=model_id
    )

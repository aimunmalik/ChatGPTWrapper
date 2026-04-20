import time
from functools import lru_cache
from typing import Any

from anna_chat.bedrock_client import BedrockClient
from anna_chat.ddb import Repository
from anna_chat.http import HttpError, authenticate, error, ok, parse_json_body
from anna_chat.logging_config import configure_logging, get_logger
from anna_chat.settings import Settings

configure_logging()
logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are ANNA Chat, an assistant for clinicians at ANNA Health, a provider of "
    "Applied Behavior Analysis (ABA) services for children with autism. Be concise, "
    "clinically accurate, and flag when something is outside your scope or would "
    "benefit from a licensed professional's judgment. Do not fabricate citations."
)


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
                user_id=user.sub, title=title, model=bedrock.model_id
            )

        history = repo.recent_turns_for_model(conversation_id=conv.conversationId)
        history.append({"role": "user", "content": user_message})

        repo.append_message(
            conversation_id=conv.conversationId,
            user_id=user.sub,
            role="user",
            content=user_message,
            model=bedrock.model_id,
        )

        bedrock_resp = bedrock.invoke(messages=history, system=SYSTEM_PROMPT)

        assistant_msg = repo.append_message(
            conversation_id=conv.conversationId,
            user_id=user.sub,
            role="assistant",
            content=bedrock_resp.text,
            input_tokens=bedrock_resp.input_tokens,
            output_tokens=bedrock_resp.output_tokens,
            model=bedrock.model_id,
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
                "model": bedrock.model_id,
                "latencyMs": latency_ms,
                "stopReason": bedrock_resp.stop_reason,
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
                "model": bedrock.model_id,
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

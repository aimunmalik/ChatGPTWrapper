from dataclasses import asdict
from functools import lru_cache
from typing import Any

from anna_chat.ddb import Repository
from anna_chat.http import HttpError, authenticate, error, ok
from anna_chat.logging_config import configure_logging, get_logger
from anna_chat.settings import Settings

configure_logging()
logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings.from_env()


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
    try:
        user = authenticate(event, _settings())
        route_key = event.get("routeKey", "")
        path_params = event.get("pathParameters") or {}
        repo = _repo()

        if route_key == "GET /conversations":
            convs = repo.list_conversations(user_id=user.sub)
            logger.info(
                "list_conversations",
                extra={"userId": user.sub, "count": len(convs)},
            )
            return ok({"conversations": [asdict(c) for c in convs]})

        if route_key == "GET /conversations/{conversationId}/messages":
            conv_id = path_params.get("conversationId", "")
            conv = repo.get_conversation(user_id=user.sub, conversation_id=conv_id)
            if not conv:
                raise HttpError(404, "conversation not found")
            msgs = repo.list_messages(conversation_id=conv_id)
            logger.info(
                "get_messages",
                extra={
                    "userId": user.sub,
                    "conversationId": conv_id,
                    "count": len(msgs),
                },
            )
            return ok(
                {
                    "conversation": asdict(conv),
                    "messages": [
                        {
                            "messageId": m.messageId,
                            "role": m.role,
                            "content": m.content,
                            "createdAt": int(m.sortKey.split("#")[0]),
                        }
                        for m in msgs
                    ],
                }
            )

        if route_key == "DELETE /conversations/{conversationId}":
            conv_id = path_params.get("conversationId", "")
            conv = repo.get_conversation(user_id=user.sub, conversation_id=conv_id)
            if not conv:
                return ok({"deleted": False}, status=204)
            repo.delete_conversation(user_id=user.sub, conversation_id=conv_id)
            logger.info(
                "delete_conversation",
                extra={"userId": user.sub, "conversationId": conv_id},
            )
            return ok({"deleted": True}, status=200)

        return error(404, "route not found")

    except HttpError as exc:
        return error(exc.status, exc.message)
    except Exception as exc:
        logger.error(
            "conversations_unhandled_error",
            extra={"errorType": type(exc).__name__},
        )
        return error(500, "internal error")

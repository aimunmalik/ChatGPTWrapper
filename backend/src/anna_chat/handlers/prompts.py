"""HTTP handler for the per-user prompt library.

Routes:
  - POST   /prompts              — create
  - GET    /prompts              — list for authenticated user
  - PUT    /prompts/{promptId}   — full replace
  - DELETE /prompts/{promptId}   — idempotent delete

Security note: prompt `title` and `body` are user-authored free text and are
treated as PHI-adjacent for audit purposes. We log metadata only
(userId, promptId, action, titleLen, bodyLen, latencyMs) — never the
title or body text itself.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from functools import lru_cache
from typing import Any

from anna_chat.http import HttpError, authenticate, error, ok, parse_json_body
from anna_chat.logging_config import configure_logging, get_logger
from anna_chat.prompts_repo import Prompt, PromptsRepo
from anna_chat.settings import Settings

configure_logging()
logger = get_logger(__name__)

TITLE_MAX = 120
BODY_MAX = 20000


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def _repo() -> PromptsRepo:
    s = _settings()
    return PromptsRepo(
        prompts_table=s.prompts_table,
        region=s.aws_region,
    )


def _prompt_response(p: Prompt) -> dict[str, Any]:
    return asdict(p)


def _validate_title_body(body: dict[str, Any]) -> tuple[str, str]:
    raw_title = body.get("title")
    raw_body = body.get("body")
    if not isinstance(raw_title, str):
        raise HttpError(400, "title is required")
    if not isinstance(raw_body, str):
        raise HttpError(400, "body is required")
    title = raw_title.strip()
    body_text = raw_body.strip()
    if not title:
        raise HttpError(400, "title must not be empty")
    if len(title) > TITLE_MAX:
        raise HttpError(400, f"title must be ≤{TITLE_MAX} chars")
    if not body_text:
        raise HttpError(400, "body must not be empty")
    if len(body_text) > BODY_MAX:
        raise HttpError(400, f"body must be ≤{BODY_MAX} chars")
    return title, body_text


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    try:
        user = authenticate(event, _settings())
        route_key = event.get("routeKey", "")
        path_params = event.get("pathParameters") or {}

        if route_key == "POST /prompts":
            return _create(event, user)

        if route_key == "GET /prompts":
            return _list(user)

        if route_key == "PUT /prompts/{promptId}":
            prompt_id = path_params.get("promptId", "")
            return _update(event, user, prompt_id)

        if route_key == "DELETE /prompts/{promptId}":
            prompt_id = path_params.get("promptId", "")
            return _delete(user, prompt_id)

        return error(404, "route not found")

    except HttpError as exc:
        logger.info(
            "prompts_http_error",
            extra={"status": exc.status, "reason": exc.message},
        )
        return error(exc.status, exc.message)
    except Exception as exc:
        logger.error(
            "prompts_unhandled_error",
            extra={"errorType": type(exc).__name__},
        )
        return error(500, "internal error")


def _create(event: dict[str, Any], user: Any) -> dict[str, Any]:
    started = time.time()
    body = parse_json_body(event)
    title, body_text = _validate_title_body(body)

    prompt = _repo().create(user_id=user.sub, title=title, body=body_text)

    logger.info(
        "prompt_created",
        extra={
            "userId": user.sub,
            "promptId": prompt.promptId,
            "action": "created",
            "titleLen": len(title),
            "bodyLen": len(body_text),
            "latencyMs": int((time.time() - started) * 1000),
        },
    )
    return ok({"prompt": _prompt_response(prompt)})


def _list(user: Any) -> dict[str, Any]:
    started = time.time()
    prompts = _repo().list_for_user(user_id=user.sub)
    logger.info(
        "prompts_list",
        extra={
            "userId": user.sub,
            "action": "list",
            "count": len(prompts),
            "latencyMs": int((time.time() - started) * 1000),
        },
    )
    return ok({"prompts": [_prompt_response(p) for p in prompts]})


def _update(event: dict[str, Any], user: Any, prompt_id: str) -> dict[str, Any]:
    if not prompt_id:
        raise HttpError(400, "promptId is required")
    started = time.time()
    body = parse_json_body(event)
    title, body_text = _validate_title_body(body)

    updated = _repo().update(
        user_id=user.sub,
        prompt_id=prompt_id,
        title=title,
        body=body_text,
    )
    if not updated:
        raise HttpError(404, "prompt not found")

    logger.info(
        "prompt_updated",
        extra={
            "userId": user.sub,
            "promptId": prompt_id,
            "action": "updated",
            "titleLen": len(title),
            "bodyLen": len(body_text),
            "latencyMs": int((time.time() - started) * 1000),
        },
    )
    return ok({"prompt": _prompt_response(updated)})


def _delete(user: Any, prompt_id: str) -> dict[str, Any]:
    if not prompt_id:
        raise HttpError(400, "promptId is required")
    started = time.time()
    existed = _repo().delete(user_id=user.sub, prompt_id=prompt_id)
    logger.info(
        "prompt_deleted",
        extra={
            "userId": user.sub,
            "promptId": prompt_id,
            "action": "deleted",
            "existed": existed,
            "latencyMs": int((time.time() - started) * 1000),
        },
    )
    if not existed:
        return ok({"deleted": False}, status=204)
    return ok({"deleted": True}, status=200)

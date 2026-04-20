"""HTTP handler for attachment lifecycle routes.

Routes:
  - POST /attachments/presigned-upload
  - GET /conversations/{conversationId}/attachments
  - DELETE /attachments/{attachmentId}
"""

from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Any

import boto3

from anna_chat.attachments_repo import AttachmentsRepo
from anna_chat.ddb import Repository
from anna_chat.extractors import ALLOWED_CONTENT_TYPES
from anna_chat.http import HttpError, authenticate, error, ok, parse_json_body
from anna_chat.logging_config import configure_logging, get_logger
from anna_chat.settings import Settings

configure_logging()
logger = get_logger(__name__)

PRESIGN_EXPIRY_SECONDS = 15 * 60
_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]")


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


@lru_cache(maxsize=1)
def _attachments_repo() -> AttachmentsRepo:
    s = _settings()
    return AttachmentsRepo(
        attachments_table=s.attachments_table,
        region=s.aws_region,
        message_ttl_days=s.message_ttl_days,
    )


@lru_cache(maxsize=1)
def _s3():
    s = _settings()
    return boto3.client("s3", region_name=s.aws_region)


def _sanitize_filename(filename: str) -> str:
    """Replace non-safe characters; cap length at 128; preserve extension."""
    name = filename or "file"
    cleaned = _FILENAME_SAFE.sub("_", name)
    if len(cleaned) <= 128:
        return cleaned
    # Try to preserve extension when truncating.
    if "." in cleaned:
        stem, dot, ext = cleaned.rpartition(".")
        keep = max(128 - len(dot) - len(ext), 1)
        return stem[:keep] + dot + ext
    return cleaned[:128]


def _attachment_response(att: Any) -> dict[str, Any]:
    return {
        "attachmentId": att.attachmentId,
        "filename": att.filename,
        "contentType": att.contentType,
        "sizeBytes": att.sizeBytes,
        "status": att.status,
        "statusMessage": att.statusMessage or None,
        "extractedPreview": att.extractedPreview or None,
        "truncated": bool(att.truncated),
        "createdAt": att.createdAt,
    }


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    try:
        user = authenticate(event, _settings())
        route_key = event.get("routeKey", "")
        path_params = event.get("pathParameters") or {}

        if route_key == "POST /attachments/presigned-upload":
            return _presigned_upload(event, user)

        if route_key == "GET /conversations/{conversationId}/attachments":
            conv_id = path_params.get("conversationId", "")
            return _list_for_conversation(user, conv_id)

        if route_key == "DELETE /attachments/{attachmentId}":
            attachment_id = path_params.get("attachmentId", "")
            return _delete(user, attachment_id)

        return error(404, "route not found")

    except HttpError as exc:
        logger.info(
            "attachments_http_error",
            extra={"status": exc.status, "reason": exc.message},
        )
        return error(exc.status, exc.message)
    except Exception:
        logger.exception("attachments_unhandled_error")
        return error(500, "internal error")


def _presigned_upload(event: dict[str, Any], user: Any) -> dict[str, Any]:
    body = parse_json_body(event)
    conversation_id = (body.get("conversationId") or "").strip()
    filename = (body.get("filename") or "").strip()
    content_type = (body.get("contentType") or "").strip()
    size_bytes = body.get("sizeBytes")

    if not conversation_id:
        raise HttpError(400, "conversationId is required")
    if not filename:
        raise HttpError(400, "filename is required")
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HttpError(400, f"unsupported contentType: {content_type}")
    if not isinstance(size_bytes, int) or size_bytes <= 0:
        raise HttpError(400, "sizeBytes must be a positive integer")

    settings = _settings()
    if size_bytes > settings.attachments_max_size_bytes:
        raise HttpError(
            400,
            f"file exceeds max size of {settings.attachments_max_size_bytes} bytes",
        )

    # Ownership check on the conversation the attachment belongs to.
    conv = _repo().get_conversation(
        user_id=user.sub, conversation_id=conversation_id
    )
    if not conv:
        raise HttpError(404, "conversation not found")

    sanitized = _sanitize_filename(filename)
    repo = _attachments_repo()

    # Generate id up-front so key and row share it.
    attachment_id = AttachmentsRepo._new_attachment_id()
    s3_key = f"attachments/{user.sub}/{attachment_id}/{sanitized}"

    att = repo.create_attachment(
        user_id=user.sub,
        conversation_id=conversation_id,
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        s3_key=s3_key,
        status="uploading",
        attachment_id=attachment_id,
    )

    presigned = _s3().generate_presigned_post(
        Bucket=settings.attachments_bucket,
        Key=s3_key,
        Fields={"Content-Type": content_type},
        Conditions=[
            {"Content-Type": content_type},
            ["content-length-range", size_bytes, size_bytes],
            ["starts-with", "$key", f"attachments/{user.sub}/{att.attachmentId}/"],
        ],
        ExpiresIn=PRESIGN_EXPIRY_SECONDS,
    )

    expires_at_ms = int((time.time() + PRESIGN_EXPIRY_SECONDS) * 1000)

    logger.info(
        "attachment_presigned",
        extra={
            "userId": user.sub,
            "conversationId": conversation_id,
            "attachmentId": att.attachmentId,
            "sizeBytes": size_bytes,
            "contentType": content_type,
        },
    )

    return ok(
        {
            "attachmentId": att.attachmentId,
            "uploadUrl": presigned["url"],
            "uploadFields": presigned["fields"],
            "expiresAt": expires_at_ms,
        }
    )


def _list_for_conversation(user: Any, conversation_id: str) -> dict[str, Any]:
    if not conversation_id:
        raise HttpError(400, "conversationId is required")
    conv = _repo().get_conversation(
        user_id=user.sub, conversation_id=conversation_id
    )
    if not conv:
        raise HttpError(404, "conversation not found")

    atts = _attachments_repo().list_for_conversation(
        conversation_id=conversation_id
    )
    logger.info(
        "attachments_list",
        extra={
            "userId": user.sub,
            "conversationId": conversation_id,
            "count": len(atts),
        },
    )
    return ok({"attachments": [_attachment_response(a) for a in atts]})


def _delete(user: Any, attachment_id: str) -> dict[str, Any]:
    if not attachment_id:
        raise HttpError(400, "attachmentId is required")
    repo = _attachments_repo()
    att = repo.get_attachment(user_id=user.sub, attachment_id=attachment_id)
    if not att:
        return ok({"deleted": False}, status=204)

    settings = _settings()
    try:
        _s3().delete_object(Bucket=settings.attachments_bucket, Key=att.s3Key)
    except Exception:
        # Best-effort: log but still remove the DDB row.
        logger.exception(
            "attachment_s3_delete_failed",
            extra={
                "userId": user.sub,
                "attachmentId": attachment_id,
                "sizeBytes": att.sizeBytes,
            },
        )

    repo.delete(user_id=user.sub, attachment_id=attachment_id)
    logger.info(
        "attachment_deleted",
        extra={
            "userId": user.sub,
            "attachmentId": attachment_id,
            "conversationId": att.conversationId,
            "sizeBytes": att.sizeBytes,
        },
    )
    return ok({"deleted": True})

"""S3 ObjectCreated handler that extracts text from uploaded attachments.

Wired to the `attachments/` prefix on the attachments bucket. Each record in the
event points at a new object; this function looks up the matching DDB row (the
`userId` and `attachmentId` are embedded in the object key), downloads up to the
configured byte cap, runs the dispatcher, and writes the result back.
"""

from __future__ import annotations

import time
import urllib.parse
from functools import lru_cache
from typing import Any

import boto3

from anna_chat.attachments_repo import AttachmentsRepo
from anna_chat.extractors import (
    ExtractionResult,
    UnsupportedContentType,
    extract,
)
from anna_chat.logging_config import configure_logging, get_logger
from anna_chat.settings import Settings

configure_logging()
logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings.from_env()


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


def _parse_key(key: str) -> tuple[str, str, str] | None:
    """Return (userId, attachmentId, filename) from an S3 key or None."""
    decoded = urllib.parse.unquote_plus(key)
    parts = decoded.split("/", 3)
    if len(parts) != 4 or parts[0] != "attachments":
        return None
    _prefix, user_id, attachment_id, filename = parts
    if not user_id or not attachment_id:
        return None
    return user_id, attachment_id, filename


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    records = event.get("Records") or []
    processed = 0
    failed = 0
    for record in records:
        try:
            _process_record(record)
            processed += 1
        except Exception as exc:
            failed += 1
            logger.error(
                "extract_record_failed",
                extra={"errorType": type(exc).__name__},
            )
    logger.info(
        "extract_batch_complete",
        extra={"processed": processed, "failed": failed},
    )
    return {"processed": processed, "failed": failed}


def _process_record(record: dict[str, Any]) -> None:
    s3_info = record.get("s3") or {}
    bucket = (s3_info.get("bucket") or {}).get("name")
    obj = s3_info.get("object") or {}
    key = obj.get("key")
    if not bucket or not key:
        logger.info("extract_skipped_no_bucket_or_key")
        return

    parsed = _parse_key(key)
    if not parsed:
        logger.info("extract_skipped_bad_key", extra={"bucket": bucket})
        return
    user_id, attachment_id, _filename = parsed

    repo = _attachments_repo()
    att = repo.get_attachment(user_id=user_id, attachment_id=attachment_id)
    if not att:
        logger.info(
            "extract_skipped_no_row",
            extra={"userId": user_id, "attachmentId": attachment_id},
        )
        return

    started = time.monotonic()
    repo.update_status(
        user_id=user_id,
        attachment_id=attachment_id,
        status="extracting",
    )

    settings = _settings()
    size_bytes = int(obj.get("size") or 0)
    max_size = settings.attachments_max_size_bytes
    if size_bytes and size_bytes > max_size:
        repo.update_status(
            user_id=user_id,
            attachment_id=attachment_id,
            status="error",
            status_message=f"file exceeds max size of {max_size} bytes",
        )
        logger.info(
            "extract_rejected_too_large",
            extra={
                "userId": user_id,
                "attachmentId": attachment_id,
                "sizeBytes": size_bytes,
            },
        )
        return

    try:
        obj_resp = _s3().get_object(Bucket=bucket, Key=key)
        data = obj_resp["Body"].read(max_size + 1)
        if len(data) > max_size:
            raise ValueError(f"object exceeds max size of {max_size} bytes")

        result: ExtractionResult = extract(
            att.contentType,
            data,
            max_text_bytes=settings.attachments_max_text_bytes,
        )
        repo.set_extraction_result(
            user_id=user_id,
            attachment_id=attachment_id,
            extracted_text=result.text,
            truncated=result.truncated,
        )
        latency_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "extract_success",
            extra={
                "userId": user_id,
                "attachmentId": attachment_id,
                "conversationId": att.conversationId,
                "sizeBytes": len(data),
                "contentType": att.contentType,
                "extractorUsed": att.contentType,
                "extractedBytes": len(result.text.encode("utf-8")),
                "truncated": result.truncated,
                "latencyMs": latency_ms,
            },
        )
    except UnsupportedContentType as exc:
        repo.update_status(
            user_id=user_id,
            attachment_id=attachment_id,
            status="error",
            status_message=f"unsupported content type: {exc}",
        )
        logger.info(
            "extract_unsupported_type",
            extra={
                "userId": user_id,
                "attachmentId": attachment_id,
                "contentType": att.contentType,
            },
        )
    except Exception as exc:  # pragma: no cover — defensive
        # Never surface the raw exception message: library errors (openpyxl,
        # python-docx) frequently quote attacker/user-supplied document
        # content, which would land in DDB and then on the UI. Emit only the
        # exception class name.
        repo.update_status(
            user_id=user_id,
            attachment_id=attachment_id,
            status="error",
            status_message=f"extraction failed: {type(exc).__name__}",
        )
        logger.error(
            "extract_failed",
            extra={
                "userId": user_id,
                "attachmentId": attachment_id,
                "contentType": att.contentType,
                "errorType": type(exc).__name__,
            },
        )

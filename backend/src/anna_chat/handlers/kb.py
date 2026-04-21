"""HTTP handler for the admin-only knowledge base routes.

Routes:
  - POST   /kb/presigned-upload
  - GET    /kb/documents
  - DELETE /kb/documents/{kbDocId}

All three require the authenticated user to be in the Cognito `admins` group.
See docs/KB_CONTRACT.md for the wire format.
"""

from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from anna_chat.http import (
    HttpError,
    authenticate,
    error,
    ok,
    parse_json_body,
    require_admin,
)
from anna_chat.kb_repo import KbRepo
from anna_chat.logging_config import configure_logging, get_logger
from anna_chat.settings import Settings

configure_logging()
logger = get_logger(__name__)

PRESIGN_EXPIRY_SECONDS = 15 * 60
# Short-lived GET URLs for the "open source" button. Deliberately tighter
# than the upload window — the URL is requested at click time and the
# browser navigates to it immediately; an admin leaking the link would
# only have 5 minutes to abuse it.
DOWNLOAD_EXPIRY_SECONDS = 5 * 60
_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]")

PDF_MIME = "application/pdf"
DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
TXT_MIME = "text/plain"
CSV_MIME = "text/csv"

ALLOWED_KB_CONTENT_TYPES: frozenset[str] = frozenset(
    {PDF_MIME, DOCX_MIME, TXT_MIME, CSV_MIME}
)
ALLOWED_SOURCE_TYPES: frozenset[str] = frozenset(
    {"research", "training", "protocol", "parent-training", "other"}
)

DOC_TITLE_MIN = 1
DOC_TITLE_MAX = 200
COLLECTION_MAX = 80
_COLLECTION_BAD = re.compile(r"[\x00-\x1f]")  # strip control chars


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def _kb_repo() -> KbRepo:
    s = _settings()
    return KbRepo(kb_table=s.kb_table, region=s.aws_region)


@lru_cache(maxsize=1)
def _s3():
    s = _settings()
    # SigV4 is required by SSE-KMS-encrypted buckets. Same fix as
    # handlers/attachments.py — boto3 defaults can produce SigV2 presigned
    # POSTs that S3 rejects with HTTP 400 when the bucket is CMK-encrypted.
    return boto3.client(
        "s3",
        region_name=s.aws_region,
        config=Config(signature_version="s3v4"),
    )


def _sanitize_filename(filename: str) -> str:
    name = filename or "file"
    cleaned = _FILENAME_SAFE.sub("_", name)
    if len(cleaned) <= 128:
        return cleaned
    if "." in cleaned:
        stem, dot, ext = cleaned.rpartition(".")
        keep = max(128 - len(dot) - len(ext), 1)
        return stem[:keep] + dot + ext
    return cleaned[:128]


def _doc_response(doc: Any) -> dict[str, Any]:
    return {
        "kbDocId": doc.kbDocId,
        "docTitle": doc.docTitle,
        "sourceType": doc.sourceType,
        "collection": getattr(doc, "collection", "") or "",
        "filename": doc.filename,
        "contentType": doc.contentType,
        "sizeBytes": doc.sizeBytes,
        "status": doc.status,
        "statusMessage": doc.statusMessage or None,
        "totalChunks": doc.totalChunks,
        "uploadedBy": doc.uploadedBy,
        "createdAt": doc.createdAt,
        "updatedAt": doc.updatedAt,
    }


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    try:
        user = authenticate(event, _settings())

        route_key = event.get("routeKey", "")
        path_params = event.get("pathParameters") or {}

        # Download is NOT admin-gated — clinicians need to open the source
        # PDF Praxis cited in the chat reply, and they are not admins. All
        # other KB admin routes remain admin-only.
        if route_key == "GET /kb/documents/{kbDocId}/download":
            kb_doc_id = path_params.get("kbDocId", "")
            return _download(user, kb_doc_id)

        require_admin(user)

        if route_key == "POST /kb/presigned-upload":
            return _presigned_upload(event, user)

        if route_key == "GET /kb/documents":
            return _list_documents(user)

        if route_key == "DELETE /kb/documents/{kbDocId}":
            kb_doc_id = path_params.get("kbDocId", "")
            return _delete(user, kb_doc_id)

        return error(404, "route not found")

    except HttpError as exc:
        logger.info(
            "kb_http_error",
            extra={"status": exc.status, "reason": exc.message},
        )
        return error(exc.status, exc.message)
    except ClientError as exc:
        # AWS error code (e.g. "AccessDeniedException") is PHI-safe — unlike
        # the message, which can echo ARNs / resource names. Log the code so
        # failures like this one are self-diagnosing.
        logger.error(
            "kb_unhandled_error",
            extra={
                "errorType": type(exc).__name__,
                "awsErrorCode": exc.response.get("Error", {}).get("Code", ""),
            },
        )
        return error(500, "internal error")
    except Exception as exc:
        logger.error(
            "kb_unhandled_error",
            extra={"errorType": type(exc).__name__},
        )
        return error(500, "internal error")


def _validate_presign_body(
    body: dict[str, Any],
) -> tuple[str, str, int, str, str, str]:
    filename = (body.get("filename") or "").strip()
    content_type = (body.get("contentType") or "").strip()
    size_bytes = body.get("sizeBytes")
    doc_title_raw = body.get("docTitle")
    source_type = (body.get("sourceType") or "").strip()
    collection_raw = body.get("collection")

    if not filename:
        raise HttpError(400, "filename is required")
    if content_type not in ALLOWED_KB_CONTENT_TYPES:
        raise HttpError(400, f"unsupported contentType: {content_type}")
    if not isinstance(size_bytes, int) or size_bytes <= 0:
        raise HttpError(400, "sizeBytes must be a positive integer")
    if source_type not in ALLOWED_SOURCE_TYPES:
        raise HttpError(400, f"unsupported sourceType: {source_type}")
    if not isinstance(doc_title_raw, str):
        raise HttpError(400, "docTitle is required")
    doc_title = doc_title_raw.strip()
    if len(doc_title) < DOC_TITLE_MIN:
        raise HttpError(400, "docTitle must not be empty")
    if len(doc_title) > DOC_TITLE_MAX:
        raise HttpError(400, f"docTitle must be ≤{DOC_TITLE_MAX} chars")

    # Collection is optional. When present: strip control chars, trim, cap.
    # Empty after normalization is treated the same as absent.
    if collection_raw is None:
        collection = ""
    elif not isinstance(collection_raw, str):
        raise HttpError(400, "collection must be a string")
    else:
        collection = _COLLECTION_BAD.sub("", collection_raw).strip()
        if len(collection) > COLLECTION_MAX:
            raise HttpError(400, f"collection must be ≤{COLLECTION_MAX} chars")

    return filename, content_type, size_bytes, doc_title, source_type, collection


def _presigned_upload(event: dict[str, Any], user: Any) -> dict[str, Any]:
    body = parse_json_body(event)
    (
        filename,
        content_type,
        size_bytes,
        doc_title,
        source_type,
        collection,
    ) = _validate_presign_body(body)

    settings = _settings()
    if size_bytes > settings.kb_max_size_bytes:
        raise HttpError(
            400,
            f"file exceeds max size of {settings.kb_max_size_bytes} bytes",
        )
    if not settings.kb_bucket or not settings.kb_table:
        raise HttpError(500, "kb storage not configured")

    sanitized = _sanitize_filename(filename)
    kb_doc_id = KbRepo._new_kb_doc_id()
    s3_key = f"kb/{kb_doc_id}/{sanitized}"

    repo = _kb_repo()
    repo.create_doc(
        user_id=user.sub,
        kb_doc_id=kb_doc_id,
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        s3_key=s3_key,
        doc_title=doc_title,
        source_type=source_type,
        collection=collection,
    )

    upper = min(
        size_bytes + (size_bytes // 10) + 65536,
        settings.kb_max_size_bytes,
    )
    presigned = _s3().generate_presigned_post(
        Bucket=settings.kb_bucket,
        Key=s3_key,
        Fields={"Content-Type": content_type},
        Conditions=[
            {"Content-Type": content_type},
            ["content-length-range", 1, upper],
            ["starts-with", "$key", f"kb/{kb_doc_id}/"],
        ],
        ExpiresIn=PRESIGN_EXPIRY_SECONDS,
    )

    expires_at_ms = int((time.time() + PRESIGN_EXPIRY_SECONDS) * 1000)

    logger.info(
        "kb_presigned",
        extra={
            "userId": user.sub,
            "kbDocId": kb_doc_id,
            "sizeBytes": size_bytes,
            "contentType": content_type,
            "sourceType": source_type,
            "hasCollection": bool(collection),
        },
    )

    return ok(
        {
            "kbDocId": kb_doc_id,
            "uploadUrl": presigned["url"],
            "uploadFields": presigned["fields"],
            "expiresAt": expires_at_ms,
        }
    )


def _download(user: Any, kb_doc_id: str) -> dict[str, Any]:
    """Return a short-lived presigned GET URL for the underlying S3 object.

    Available to any authenticated user — source pills on chat replies
    need to be openable by clinicians, not just admins. We log every
    download for the HIPAA access audit trail.
    """
    if not kb_doc_id:
        raise HttpError(400, "kbDocId is required")
    repo = _kb_repo()
    doc = repo.get_doc(kb_doc_id)
    if not doc:
        raise HttpError(404, "document not found")

    settings = _settings()
    if not settings.kb_bucket:
        raise HttpError(500, "kb storage not configured")

    # Force the browser's PDF viewer / download UX to use the original
    # filename rather than the sanitized S3 key tail.
    response_content_disposition = f'inline; filename="{doc.filename}"'
    url = _s3().generate_presigned_url(
        "get_object",
        Params={
            "Bucket": settings.kb_bucket,
            "Key": doc.s3Key,
            "ResponseContentDisposition": response_content_disposition,
            "ResponseContentType": doc.contentType,
        },
        ExpiresIn=DOWNLOAD_EXPIRY_SECONDS,
    )
    expires_at_ms = int((time.time() + DOWNLOAD_EXPIRY_SECONDS) * 1000)

    logger.info(
        "kb_download",
        extra={
            "userId": user.sub,
            "kbDocId": kb_doc_id,
            "sourceType": doc.sourceType,
        },
    )

    return ok(
        {
            "kbDocId": kb_doc_id,
            "url": url,
            "expiresAt": expires_at_ms,
            "filename": doc.filename,
            "docTitle": doc.docTitle,
            "contentType": doc.contentType,
        }
    )


def _list_documents(user: Any) -> dict[str, Any]:
    docs = _kb_repo().list_docs()
    logger.info(
        "kb_list_documents",
        extra={"userId": user.sub, "count": len(docs)},
    )
    # Sort newest-first for the UI — contract doesn't mandate an order but
    # that's what every admin dashboard expects.
    docs_sorted = sorted(docs, key=lambda d: d.createdAt, reverse=True)
    return ok({"documents": [_doc_response(d) for d in docs_sorted]})


def _delete(user: Any, kb_doc_id: str) -> dict[str, Any]:
    if not kb_doc_id:
        raise HttpError(400, "kbDocId is required")
    repo = _kb_repo()
    doc = repo.get_doc(kb_doc_id)
    if not doc:
        return ok({"deleted": False}, status=204)

    settings = _settings()
    try:
        _s3().delete_object(Bucket=settings.kb_bucket, Key=doc.s3Key)
    except Exception as exc:
        # Best-effort — remove the DDB rows regardless so the table and
        # bucket can't diverge permanently. S3 has its own retention; we'll
        # rely on object expiry for any leftovers.
        logger.error(
            "kb_s3_delete_failed",
            extra={
                "userId": user.sub,
                "kbDocId": kb_doc_id,
                "errorType": type(exc).__name__,
            },
        )

    repo.delete_doc(kb_doc_id)

    # Drop the in-memory retrieval cache so the next chat turn doesn't score
    # against stale chunks from the just-deleted doc.
    try:
        from anna_chat import kb_retrieve

        kb_retrieve.invalidate_cache()
    except Exception:  # pragma: no cover — cache invalidation is best-effort
        pass

    logger.info(
        "kb_deleted",
        extra={
            "userId": user.sub,
            "kbDocId": kb_doc_id,
            "totalChunks": doc.totalChunks,
        },
    )
    return ok({"deleted": True})

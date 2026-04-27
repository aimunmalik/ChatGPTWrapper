"""HTTP handler for the translation-job routes.

Routes:
  - POST /translate/jobs
  - GET  /translate/jobs
  - GET  /translate/jobs/{jobId}
  - GET  /translate/jobs/{jobId}/download/{format}

All routes are JWT-authenticated. None are admin-gated — every signed-in
user can translate documents they own. POST async-invokes the worker
Lambda (`InvocationType=Event`), so the API GW request returns within
the 30s integration window regardless of how long translation takes.

See docs/TRANSLATE_CONTRACT.md for the wire format.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from anna_chat.attachments_repo import AttachmentsRepo
from anna_chat.http import (
    HttpError,
    authenticate,
    error,
    ok,
    parse_json_body,
)
from anna_chat.jobs_repo import JobsRepo, TranslateJob
from anna_chat.logging_config import configure_logging, get_logger
from anna_chat.settings import Settings

configure_logging()
logger = get_logger(__name__)

DOWNLOAD_EXPIRY_SECONDS = 5 * 60
DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
PDF_MIME = "application/pdf"

# Languages supported by V1 (per contract). Adding a language is a one-line
# config change here.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "es": "Spanish",
    "zh": "Mandarin (Simplified Chinese)",
    "vi": "Vietnamese",
    "tl": "Tagalog",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "fr": "French",
    "pt": "Portuguese (Brazilian)",
    "hi": "Hindi",
    "de": "German",
    "ja": "Japanese",
    "en": "English",
}

VALID_FORMATS: frozenset[str] = frozenset({"docx", "pdf"})


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def _jobs_repo() -> JobsRepo:
    s = _settings()
    return JobsRepo(jobs_table=s.jobs_table, region=s.aws_region)


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
    # SigV4 required for SSE-KMS-encrypted buckets — same fix as kb.py.
    return boto3.client(
        "s3",
        region_name=s.aws_region,
        config=Config(signature_version="s3v4"),
    )


@lru_cache(maxsize=1)
def _lambda():
    s = _settings()
    return boto3.client("lambda", region_name=s.aws_region)


def _job_response(job: TranslateJob) -> dict[str, Any]:
    download_docx_url: str | None = None
    download_pdf_url: str | None = None
    if job.status == "ready" and job.outputDocxKey and job.outputPdfKey:
        download_docx_url = f"/translate/jobs/{job.jobId}/download/docx"
        download_pdf_url = f"/translate/jobs/{job.jobId}/download/pdf"
    return {
        "jobId": job.jobId,
        "userId": job.userId,
        "status": job.status,
        "statusMessage": job.statusMessage or None,
        "sourceAttachmentId": job.sourceAttachmentId,
        "sourceFilename": job.sourceFilename,
        "sourceContentType": job.sourceContentType,
        "targetLanguage": job.targetLanguage,
        "targetLanguageLabel": job.targetLanguageLabel,
        "createdAt": job.createdAt,
        "updatedAt": job.updatedAt,
        "inputTokensApprox": job.inputTokensApprox,
        "outputTokensApprox": job.outputTokensApprox,
        "downloadDocxUrl": download_docx_url,
        "downloadPdfUrl": download_pdf_url,
    }


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    try:
        user = authenticate(event, _settings())
        route_key = event.get("routeKey", "")
        path_params = event.get("pathParameters") or {}

        if route_key == "POST /translate/jobs":
            return _create_job(event, user)

        if route_key == "GET /translate/jobs":
            return _list_jobs(user)

        if route_key == "GET /translate/jobs/{jobId}":
            return _get_job(user, path_params.get("jobId", ""))

        if route_key == "GET /translate/jobs/{jobId}/download/{format}":
            return _download(
                user,
                path_params.get("jobId", ""),
                path_params.get("format", ""),
            )

        return error(404, "route not found")

    except HttpError as exc:
        logger.info(
            "translate_http_error",
            extra={"status": exc.status, "reason": exc.message},
        )
        return error(exc.status, exc.message)
    except ClientError as exc:
        logger.error(
            "translate_unhandled_error",
            extra={
                "errorType": type(exc).__name__,
                "awsErrorCode": exc.response.get("Error", {}).get("Code", ""),
            },
        )
        return error(500, "internal error")
    except Exception as exc:
        logger.error(
            "translate_unhandled_error",
            extra={"errorType": type(exc).__name__},
        )
        return error(500, "internal error")


def _validate_create_body(body: dict[str, Any]) -> tuple[str, str, str]:
    attachment_id = (body.get("attachmentId") or "").strip()
    target_language = (body.get("targetLanguage") or "").strip()
    if not attachment_id:
        raise HttpError(400, "attachmentId is required")
    if not target_language:
        raise HttpError(400, "targetLanguage is required")
    label = SUPPORTED_LANGUAGES.get(target_language)
    if label is None:
        raise HttpError(
            400, f"unsupported targetLanguage: {target_language}"
        )
    return attachment_id, target_language, label


def _create_job(event: dict[str, Any], user: Any) -> dict[str, Any]:
    body = parse_json_body(event)
    attachment_id, target_language, target_label = _validate_create_body(body)

    settings = _settings()
    if not settings.jobs_table:
        raise HttpError(500, "translate jobs storage not configured")
    if not settings.translate_worker_function_name:
        raise HttpError(500, "translate worker not configured")

    # Ownership + readiness check — `get_attachment` is keyed by `(userId,
    # attachmentId)` so a missing row could mean either "doesn't exist" or
    # "owned by someone else". Per the contract we collapse both to 404 to
    # avoid existence enumeration. The status check is a separate 404
    # because the user can fix it (wait for extraction).
    att = _attachments_repo().get_attachment(
        user_id=user.sub, attachment_id=attachment_id
    )
    if not att:
        raise HttpError(404, "attachment not found")
    if att.status != "ready" or not att.extractedText:
        raise HttpError(
            404, "attachment not yet ready for translation"
        )

    repo = _jobs_repo()
    job = repo.create_job(
        user_id=user.sub,
        source_attachment_id=attachment_id,
        source_filename=att.filename,
        source_content_type=att.contentType,
        target_language=target_language,
        target_language_label=target_label,
    )

    # Async-invoke the worker. `Event` returns 202 from Lambda immediately;
    # boto3 raises ClientError on access denied / function-not-found, which
    # is caught by the top-level handler.
    payload = json.dumps({"jobId": job.jobId, "userId": user.sub})
    _lambda().invoke(
        FunctionName=settings.translate_worker_function_name,
        InvocationType="Event",
        Payload=payload.encode("utf-8"),
    )

    logger.info(
        "translate_job_created",
        extra={
            "userId": user.sub,
            "jobId": job.jobId,
            "sourceAttachmentId": attachment_id,
            "sourceContentType": att.contentType,
            "sourceSizeBytes": att.sizeBytes,
            "targetLanguage": target_language,
        },
    )

    return ok(
        {
            "jobId": job.jobId,
            "status": job.status,
            "sourceFilename": job.sourceFilename,
            "targetLanguageLabel": job.targetLanguageLabel,
            "createdAt": job.createdAt,
        },
        status=201,
    )


def _get_job(user: Any, job_id: str) -> dict[str, Any]:
    if not job_id:
        raise HttpError(400, "jobId is required")
    job = _jobs_repo().get_job(user_id=user.sub, job_id=job_id)
    if not job:
        # Deliberately collapse "doesn't exist" and "owned by someone else"
        # into 404 to avoid existence enumeration (per contract).
        raise HttpError(404, "job not found")
    return ok(_job_response(job))


def _list_jobs(user: Any) -> dict[str, Any]:
    jobs = _jobs_repo().list_jobs(user_id=user.sub, limit=50)
    logger.info(
        "translate_list_jobs",
        extra={"userId": user.sub, "count": len(jobs)},
    )
    return ok({"jobs": [_job_response(j) for j in jobs]})


def _download(user: Any, job_id: str, fmt: str) -> dict[str, Any]:
    if not job_id:
        raise HttpError(400, "jobId is required")
    if fmt not in VALID_FORMATS:
        raise HttpError(400, f"invalid format: {fmt}")

    job = _jobs_repo().get_job(user_id=user.sub, job_id=job_id)
    if not job:
        raise HttpError(404, "job not found")
    if job.status != "ready":
        raise HttpError(404, "job not yet ready")

    settings = _settings()
    if not settings.attachments_bucket:
        raise HttpError(500, "translation storage not configured")

    if fmt == "docx":
        s3_key = job.outputDocxKey
        content_type = DOCX_MIME
        extension = "docx"
    else:
        s3_key = job.outputPdfKey
        content_type = PDF_MIME
        extension = "pdf"

    if not s3_key:
        raise HttpError(404, "output file missing")

    download_filename = _download_filename(
        job.sourceFilename, job.targetLanguage, extension
    )
    response_content_disposition = (
        f'attachment; filename="{download_filename}"'
    )
    url = _s3().generate_presigned_url(
        "get_object",
        Params={
            "Bucket": settings.attachments_bucket,
            "Key": s3_key,
            "ResponseContentDisposition": response_content_disposition,
            "ResponseContentType": content_type,
        },
        ExpiresIn=DOWNLOAD_EXPIRY_SECONDS,
    )
    expires_at_ms = int((time.time() + DOWNLOAD_EXPIRY_SECONDS) * 1000)

    logger.info(
        "translate_download",
        extra={
            "userId": user.sub,
            "jobId": job_id,
            "format": fmt,
        },
    )

    return ok(
        {
            "url": url,
            "expiresAt": expires_at_ms,
            "filename": download_filename,
            "contentType": content_type,
        }
    )


def _download_filename(source_filename: str, language_code: str, ext: str) -> str:
    """Produce `{stem}_{lang}.{ext}` from the original filename.

    Strips the original extension and any path components for the
    `Content-Disposition` header. Browser hint only — actual S3 key is
    set by the worker.
    """
    name = source_filename.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    safe_stem = stem.replace('"', "").strip() or "translation"
    return f"{safe_stem}_{language_code}.{ext}"

r"""Async worker Lambda for the translation pipeline.

Invoked by `handlers/translate._create_job` with `InvocationType=Event`.
Drives the contract's state machine:

  pending -> extracting -> chunking -> translating -> formatting -> ready
                                                                 \-> error

Source text is the attachment row's already-extracted `extractedText`
field — no re-extraction. Renders .docx and .pdf via document_formatters,
uploads both to the existing attachments bucket under the
`translations/{jobId}/` prefix, then sets the keys on the job row.

PHI rules: never log chunk text, source text, or translation output.
Counts and IDs only.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import boto3

from anna_chat.attachments_repo import AttachmentsRepo
from anna_chat.bedrock_client import ASYNC_READ_TIMEOUT, BedrockClient
from anna_chat.chunking import approx_token_count
from anna_chat.document_formatters import build_docx, build_pdf
from anna_chat.jobs_repo import (
    STATUS_CHUNKING,
    STATUS_ERROR,
    STATUS_EXTRACTING,
    STATUS_FORMATTING,
    STATUS_TRANSLATING,
    JobsRepo,
)
from anna_chat.logging_config import configure_logging, get_logger
from anna_chat.settings import Settings
from anna_chat.translate import translate_text

configure_logging()
logger = get_logger(__name__)

DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
PDF_MIME = "application/pdf"

# Sonnet's effective context window is ~200K tokens; over that we fail-fast
# with `TooLarge` rather than truncating silently.
MAX_INPUT_TOKENS = 200_000


class SourceNotReady(Exception):
    """Source attachment hasn't finished extraction yet."""


class SourceNotFound(Exception):
    """Source attachment row is missing or has no extracted text."""


class TooLarge(Exception):
    """Source text exceeds the per-job token budget."""


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
def _bedrock() -> BedrockClient:
    """Bedrock client tuned for the LONG-running translation worker.

    Uses ASYNC_READ_TIMEOUT (180s) instead of the 25s cap the chat handler
    inherits — translation chunks can legitimately take 60s+ to generate
    3000+ output tokens, and this Lambda has a 15-min budget regardless.
    The chat handler keeps the 25s cap so it never blows past API GW's 30s.
    """
    s = _settings()
    return BedrockClient(
        region=s.aws_region,
        model_id=s.bedrock_model_id,
        read_timeout=ASYNC_READ_TIMEOUT,
    )


@lru_cache(maxsize=1)
def _s3():
    s = _settings()
    return boto3.client("s3", region_name=s.aws_region)


def _output_filename(source_filename: str, language_code: str, ext: str) -> str:
    """Produce `{stem}_{lang}.{ext}` for the S3 key tail."""
    name = source_filename.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    safe_stem = stem.strip() or "translation"
    return f"{safe_stem}_{language_code}.{ext}"


def _put_object(*, bucket: str, key: str, body: bytes, content_type: str) -> None:
    _s3().put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
        # Defense-in-depth: the bucket already enforces SSE-KMS, but stating
        # it on the request makes the intent explicit and protects against
        # an inherited-policy regression.
        ServerSideEncryption="aws:kms",
    )


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    job_id = (event or {}).get("jobId")
    user_id = (event or {}).get("userId")
    if not job_id or not user_id:
        logger.error(
            "translate_worker_bad_event",
            extra={"hasJobId": bool(job_id), "hasUserId": bool(user_id)},
        )
        return {"ok": False, "reason": "missing jobId or userId"}

    repo = _jobs_repo()
    job = repo.get_job(user_id=user_id, job_id=job_id)
    if not job:
        logger.error(
            "translate_worker_job_missing",
            extra={"userId": user_id, "jobId": job_id},
        )
        return {"ok": False, "reason": "job not found"}

    settings = _settings()

    try:
        # --- extracting (load source from attachments table) ---
        repo.update_status(
            user_id=user_id, job_id=job_id, status=STATUS_EXTRACTING
        )
        att = _attachments_repo().get_attachment(
            user_id=user_id, attachment_id=job.sourceAttachmentId
        )
        if not att:
            raise SourceNotFound(job.sourceAttachmentId)
        if att.status != "ready":
            raise SourceNotReady(att.status)
        source_text = att.extractedText or ""
        if not source_text.strip():
            raise SourceNotFound("no extracted text")

        # --- chunking (token budget gate) ---
        repo.update_status(
            user_id=user_id, job_id=job_id, status=STATUS_CHUNKING
        )
        input_tokens_approx = approx_token_count(source_text)
        if input_tokens_approx > MAX_INPUT_TOKENS:
            raise TooLarge(f"{input_tokens_approx} > {MAX_INPUT_TOKENS}")

        # --- translating ---
        repo.update_status(
            user_id=user_id,
            job_id=job_id,
            status=STATUS_TRANSLATING,
            input_tokens_approx=input_tokens_approx,
        )
        result = translate_text(
            source_text,
            job.targetLanguageLabel,
            bedrock=_bedrock(),
        )

        # --- formatting (build .docx + .pdf, upload to S3) ---
        repo.update_status(
            user_id=user_id,
            job_id=job_id,
            status=STATUS_FORMATTING,
            output_tokens_approx=result.output_tokens,
        )
        title = f"{job.sourceFilename} ({job.targetLanguageLabel})"
        docx_bytes = build_docx(title, result.text, job.targetLanguage)
        pdf_bytes = build_pdf(title, result.text, job.targetLanguage)

        docx_filename = _output_filename(
            job.sourceFilename, job.targetLanguage, "docx"
        )
        pdf_filename = _output_filename(
            job.sourceFilename, job.targetLanguage, "pdf"
        )
        docx_key = f"translations/{job_id}/{docx_filename}"
        pdf_key = f"translations/{job_id}/{pdf_filename}"

        if not settings.attachments_bucket:
            raise RuntimeError("attachments_bucket not configured")

        _put_object(
            bucket=settings.attachments_bucket,
            key=docx_key,
            body=docx_bytes,
            content_type=DOCX_MIME,
        )
        _put_object(
            bucket=settings.attachments_bucket,
            key=pdf_key,
            body=pdf_bytes,
            content_type=PDF_MIME,
        )

        # --- ready ---
        repo.set_outputs(
            user_id=user_id,
            job_id=job_id,
            output_docx_key=docx_key,
            output_pdf_key=pdf_key,
            output_tokens_approx=result.output_tokens,
        )
        logger.info(
            "translate_job_complete",
            extra={
                "userId": user_id,
                "jobId": job_id,
                "targetLanguage": job.targetLanguage,
                "inputTokensApprox": input_tokens_approx,
                "outputTokensApprox": result.output_tokens,
                "docxBytes": len(docx_bytes),
                "pdfBytes": len(pdf_bytes),
            },
        )
        return {"ok": True, "jobId": job_id}

    except Exception as exc:
        # `type(exc).__name__` is PHI-safe; the message is not (it can echo
        # the source filename or library-formatted snippets). Match kb_ingest
        # behavior — record the type as `statusMessage`, log the type only.
        error_type = type(exc).__name__
        repo.update_status(
            user_id=user_id,
            job_id=job_id,
            status=STATUS_ERROR,
            status_message=error_type,
        )
        logger.error(
            "translate_failed",
            extra={
                "userId": user_id,
                "jobId": job_id,
                "errorType": error_type,
                "stage": _stage_for_status(job.status),
            },
        )
        return {"ok": False, "jobId": job_id, "errorType": error_type}


def _stage_for_status(status: str) -> str:
    """Map a job status to the audit-log `stage` field."""
    return status or "unknown"

"""S3 ObjectCreated handler for knowledge base documents.

Key shape: `kb/{kbDocId}/{sanitizedFilename}`. Triggered by the S3 event
notification wired from the KB bucket. The flow is a straight state machine:

  uploading -> extracting -> chunking -> embedding -> ready
                    |            |            |
                    +--- any of these can transition to `error` ---+

On `error`, `statusMessage` is the exception class name (never the message
itself — library exceptions often quote user-supplied content).

See docs/KB_CONTRACT.md for the broader design.
"""

from __future__ import annotations

import random
import time
import urllib.parse
from functools import lru_cache
from typing import Any

import boto3
from botocore.exceptions import ClientError

from anna_chat.chunking import chunk_text
from anna_chat.embeddings import EmbeddingsClient
from anna_chat.extractors import (
    CSV_MIME,
    DOCX_MIME,
    PDF_MIME,
    TXT_MIME,
    UnsupportedContentType,
    extract,
)
from anna_chat.kb_repo import ChunkRecord, KbRepo
from anna_chat.logging_config import configure_logging, get_logger
from anna_chat.settings import Settings

configure_logging()
logger = get_logger(__name__)

MAX_EMBED_RETRIES = 3
EMBED_BACKOFF_BASE = 0.5  # seconds — doubles each retry, plus jitter

# KB extraction accepts a narrower set than attachments (no XLSX, no images):
# plain text that can be RAG'd directly. See KB contract.
KB_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {PDF_MIME, DOCX_MIME, TXT_MIME, CSV_MIME}
)


class UnsupportedKbContentType(Exception):
    """Raised when a KB upload is a type we don't ingest (e.g. XLSX, image)."""


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def _kb_repo() -> KbRepo:
    s = _settings()
    return KbRepo(kb_table=s.kb_table, region=s.aws_region)


@lru_cache(maxsize=1)
def _embeddings() -> EmbeddingsClient:
    s = _settings()
    return EmbeddingsClient(region=s.aws_region)


@lru_cache(maxsize=1)
def _s3():
    s = _settings()
    return boto3.client("s3", region_name=s.aws_region)


def _parse_key(key: str) -> tuple[str, str] | None:
    """Return (kbDocId, filename) from an S3 key shaped `kb/{id}/{file}`."""
    decoded = urllib.parse.unquote_plus(key)
    parts = decoded.split("/", 2)
    if len(parts) != 3 or parts[0] != "kb":
        return None
    _prefix, kb_doc_id, filename = parts
    if not kb_doc_id or not filename:
        return None
    return kb_doc_id, filename


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
                "kb_ingest_record_failed",
                extra={"errorType": type(exc).__name__},
            )
    logger.info(
        "kb_ingest_batch_complete",
        extra={"processed": processed, "failed": failed},
    )
    return {"processed": processed, "failed": failed}


def _embed_with_retry(embeddings: EmbeddingsClient, text: str) -> list[float]:
    """Call `embeddings.embed` with exponential backoff on Bedrock throttling."""
    last_exc: Exception | None = None
    for attempt in range(MAX_EMBED_RETRIES):
        try:
            return embeddings.embed(text)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in {"ThrottlingException", "TooManyRequestsException"}:
                raise
            last_exc = exc
            delay = EMBED_BACKOFF_BASE * (2**attempt) + random.uniform(0, 0.25)
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _process_record(record: dict[str, Any]) -> None:
    s3_info = record.get("s3") or {}
    bucket = (s3_info.get("bucket") or {}).get("name")
    obj = s3_info.get("object") or {}
    key = obj.get("key")
    if not bucket or not key:
        logger.info("kb_ingest_skipped_no_bucket_or_key")
        return

    parsed = _parse_key(key)
    if not parsed:
        logger.info("kb_ingest_skipped_bad_key", extra={"bucket": bucket})
        return
    kb_doc_id, _filename = parsed

    repo = _kb_repo()
    doc = repo.get_doc(kb_doc_id)
    if not doc:
        logger.info(
            "kb_ingest_skipped_no_row",
            extra={"kbDocId": kb_doc_id},
        )
        return

    started = time.monotonic()

    try:
        if doc.contentType not in KB_ALLOWED_CONTENT_TYPES:
            raise UnsupportedKbContentType(doc.contentType)

        # --- extracting ---
        repo.update_doc_status(kb_doc_id=kb_doc_id, status="extracting")

        settings = _settings()
        max_size = settings.kb_max_size_bytes
        size_bytes = int(obj.get("size") or 0)
        if size_bytes and size_bytes > max_size:
            raise ValueError(f"object exceeds max size of {max_size} bytes")

        obj_resp = _s3().get_object(Bucket=bucket, Key=key)
        data = obj_resp["Body"].read(max_size + 1)
        if len(data) > max_size:
            raise ValueError(f"object exceeds max size of {max_size} bytes")

        # KB text cap — we don't truncate to 500 KB like attachments because we
        # chunk downstream and can handle much more. Cap at 5 MB of extracted
        # text as a safety net.
        result = extract(doc.contentType, data, max_text_bytes=5 * 1024 * 1024)
        extracted = result.text

        # --- chunking ---
        repo.update_doc_status(kb_doc_id=kb_doc_id, status="chunking")
        windows = chunk_text(extracted)
        if not windows:
            repo.update_doc_status(
                kb_doc_id=kb_doc_id,
                status="ready",
                status_message="",
                total_chunks=0,
            )
            logger.info(
                "kb_ingest_empty_document",
                extra={
                    "kbDocId": kb_doc_id,
                    "totalChunks": 0,
                    "latencyMs": int((time.monotonic() - started) * 1000),
                },
            )
            return

        # --- embedding ---
        repo.update_doc_status(kb_doc_id=kb_doc_id, status="embedding")
        embeddings = _embeddings()
        tokens_embedded = 0
        chunk_records: list[ChunkRecord] = []
        for idx, window in enumerate(windows):
            vec = _embed_with_retry(embeddings, window.text)
            tokens_embedded += window.approx_tokens
            chunk_records.append(
                ChunkRecord(
                    chunkIdx=idx,
                    chunkText=window.text,
                    chunkTokens=window.approx_tokens,
                    embedding=vec,
                    pageNumber=window.page_number,
                    sectionTitle=window.section_title,
                )
            )

        repo.write_chunks(kb_doc_id, chunk_records)
        repo.update_doc_status(
            kb_doc_id=kb_doc_id,
            status="ready",
            status_message="",
            total_chunks=len(chunk_records),
        )

        latency_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "kb_ingest_success",
            extra={
                "kbDocId": kb_doc_id,
                "sourceType": doc.sourceType,
                "totalChunks": len(chunk_records),
                "tokensEmbedded": tokens_embedded,
                "sizeBytes": len(data),
                "latencyMs": latency_ms,
            },
        )
    except UnsupportedKbContentType as exc:
        repo.update_doc_status(
            kb_doc_id=kb_doc_id,
            status="error",
            status_message=type(exc).__name__,
        )
        logger.info(
            "kb_ingest_unsupported_type",
            extra={"kbDocId": kb_doc_id, "contentType": doc.contentType},
        )
    except UnsupportedContentType as exc:
        repo.update_doc_status(
            kb_doc_id=kb_doc_id,
            status="error",
            status_message=type(exc).__name__,
        )
        logger.info(
            "kb_ingest_unsupported_type",
            extra={"kbDocId": kb_doc_id, "contentType": doc.contentType},
        )
    except Exception as exc:
        repo.update_doc_status(
            kb_doc_id=kb_doc_id,
            status="error",
            status_message=type(exc).__name__,
        )
        logger.error(
            "kb_ingest_failed",
            extra={
                "kbDocId": kb_doc_id,
                "errorType": type(exc).__name__,
            },
        )

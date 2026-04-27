"""DynamoDB repository for the translation jobs table.

Matches the contract in docs/TRANSLATE_CONTRACT.md. The jobs table is keyed
by `(userId, jobId)` — partition by Cognito sub so a Query on the user can
list every job they own with no GSI required.

Style mirrors anna_chat.kb_repo.KbRepo and anna_chat.attachments_repo —
boto3 resource API, dataclass for the row, Decimal-aware on read, defensive
float-to-Decimal conversion on writes (no floats today, but the cost of
defending in depth is one helper call).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key

from anna_chat.ddb import _floats_to_decimal

JOB_TYPE_TRANSLATE = "translate"

# Statuses defined by the contract's state machine. Worker drives transitions
# pending -> extracting -> chunking -> translating -> formatting -> ready,
# or any stage -> error.
STATUS_PENDING = "pending"
STATUS_EXTRACTING = "extracting"
STATUS_CHUNKING = "chunking"
STATUS_TRANSLATING = "translating"
STATUS_FORMATTING = "formatting"
STATUS_READY = "ready"
STATUS_ERROR = "error"

JOB_TTL_DAYS = 7


@dataclass
class TranslateJob:
    userId: str
    jobId: str
    jobType: str
    status: str
    sourceAttachmentId: str
    sourceFilename: str
    sourceContentType: str
    targetLanguage: str
    targetLanguageLabel: str
    createdAt: int
    updatedAt: int
    ttl: int
    statusMessage: str = ""
    outputDocxKey: str = ""
    outputPdfKey: str = ""
    inputTokensApprox: int = 0
    outputTokensApprox: int = 0


class JobsRepo:
    """DynamoDB repository for the `anna-chat-{env}-jobs` table."""

    def __init__(self, *, jobs_table: str, region: str) -> None:
        ddb = boto3.resource("dynamodb", region_name=region)
        self._table = ddb.Table(jobs_table)
        self._ttl_seconds = JOB_TTL_DAYS * 86400

    # ---------- helpers ----------

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _new_job_id() -> str:
        return f"tj_{uuid.uuid4().hex[:16]}"

    @staticmethod
    def _to_job(item: dict[str, Any]) -> TranslateJob:
        """Hydrate a DDB item into a `TranslateJob`, casting Decimals to ints."""
        known = {
            "userId",
            "jobId",
            "jobType",
            "status",
            "statusMessage",
            "sourceAttachmentId",
            "sourceFilename",
            "sourceContentType",
            "targetLanguage",
            "targetLanguageLabel",
            "outputDocxKey",
            "outputPdfKey",
            "inputTokensApprox",
            "outputTokensApprox",
            "createdAt",
            "updatedAt",
            "ttl",
        }
        kwargs: dict[str, Any] = {k: item[k] for k in known if k in item}
        for numfield in (
            "inputTokensApprox",
            "outputTokensApprox",
            "createdAt",
            "updatedAt",
            "ttl",
        ):
            if numfield in kwargs and kwargs[numfield] is not None:
                kwargs[numfield] = int(kwargs[numfield])
        # Provide defaults for fields the dataclass requires but DDB may have
        # dropped (e.g. older rows pre-feature). Defending against missing
        # fields keeps reads safe even if the schema evolves.
        kwargs.setdefault("statusMessage", "")
        kwargs.setdefault("outputDocxKey", "")
        kwargs.setdefault("outputPdfKey", "")
        kwargs.setdefault("inputTokensApprox", 0)
        kwargs.setdefault("outputTokensApprox", 0)
        kwargs.setdefault("ttl", 0)
        return TranslateJob(**kwargs)

    # ---------- CRUD ----------

    def create_job(
        self,
        *,
        user_id: str,
        source_attachment_id: str,
        source_filename: str,
        source_content_type: str,
        target_language: str,
        target_language_label: str,
        job_id: str | None = None,
    ) -> TranslateJob:
        """Insert a new translation job row in `pending` state."""
        now_ms = self._now_ms()
        job = TranslateJob(
            userId=user_id,
            jobId=job_id or self._new_job_id(),
            jobType=JOB_TYPE_TRANSLATE,
            status=STATUS_PENDING,
            sourceAttachmentId=source_attachment_id,
            sourceFilename=source_filename,
            sourceContentType=source_content_type,
            targetLanguage=target_language,
            targetLanguageLabel=target_language_label,
            createdAt=now_ms,
            updatedAt=now_ms,
            ttl=int(time.time()) + self._ttl_seconds,
        )
        item = asdict(job)
        # Drop empty optional output keys at write time so the row stays clean
        # until the worker fills them in. Same pattern as KbRepo with empty
        # tags / collection.
        for empty_field in ("outputDocxKey", "outputPdfKey"):
            if not item.get(empty_field):
                item.pop(empty_field, None)
        self._table.put_item(Item=_floats_to_decimal(item))
        return job

    def get_job(self, *, user_id: str, job_id: str) -> TranslateJob | None:
        resp = self._table.get_item(Key={"userId": user_id, "jobId": job_id})
        item = resp.get("Item")
        return self._to_job(item) if item else None

    def list_jobs(
        self, *, user_id: str, limit: int = 50
    ) -> list[TranslateJob]:
        """Return the caller's jobs newest-first, capped at `limit`."""
        resp = self._table.query(
            KeyConditionExpression=Key("userId").eq(user_id),
            Limit=limit,
            ScanIndexForward=False,
        )
        return [self._to_job(item) for item in resp.get("Items", [])]

    def update_status(
        self,
        *,
        user_id: str,
        job_id: str,
        status: str,
        status_message: str | None = None,
        input_tokens_approx: int | None = None,
        output_tokens_approx: int | None = None,
    ) -> None:
        """Update the job's status and optional bookkeeping fields.

        `updatedAt` always bumps so the UI poll sees fresh timestamps.
        """
        expr_parts = ["#s = :s", "updatedAt = :u"]
        names = {"#s": "status"}
        values: dict[str, Any] = {":s": status, ":u": self._now_ms()}
        if status_message is not None:
            expr_parts.append("statusMessage = :m")
            values[":m"] = status_message
        if input_tokens_approx is not None:
            expr_parts.append("inputTokensApprox = :it")
            values[":it"] = int(input_tokens_approx)
        if output_tokens_approx is not None:
            expr_parts.append("outputTokensApprox = :ot")
            values[":ot"] = int(output_tokens_approx)
        expr = "SET " + ", ".join(expr_parts)
        self._table.update_item(
            Key={"userId": user_id, "jobId": job_id},
            UpdateExpression=expr,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=_floats_to_decimal(values),
        )

    def set_outputs(
        self,
        *,
        user_id: str,
        job_id: str,
        output_docx_key: str,
        output_pdf_key: str,
        output_tokens_approx: int | None = None,
    ) -> None:
        """Mark the job `ready` and record the .docx/.pdf S3 keys."""
        expr_parts = [
            "#s = :s",
            "updatedAt = :u",
            "statusMessage = :m",
            "outputDocxKey = :dk",
            "outputPdfKey = :pk",
        ]
        names = {"#s": "status"}
        values: dict[str, Any] = {
            ":s": STATUS_READY,
            ":u": self._now_ms(),
            ":m": "",
            ":dk": output_docx_key,
            ":pk": output_pdf_key,
        }
        if output_tokens_approx is not None:
            expr_parts.append("outputTokensApprox = :ot")
            values[":ot"] = int(output_tokens_approx)
        expr = "SET " + ", ".join(expr_parts)
        self._table.update_item(
            Key={"userId": user_id, "jobId": job_id},
            UpdateExpression=expr,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=_floats_to_decimal(values),
        )

"""DynamoDB repository for the knowledge base table.

Matches the contract in docs/KB_CONTRACT.md. The KB table is a single-table
design keyed by `(kbDocId, sk)`. A document is represented by one META item
(sk = "META") plus one chunk item per chunk (sk = "chunk#NNNN" zero-padded).

Style mirrors anna_chat.attachments_repo.AttachmentsRepo and
anna_chat.prompts_repo.PromptsRepo — resource API, Key conditions, Decimal
normalization on read.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.conditions import Attr, Key

META_SK = "META"
CHUNK_SK_PREFIX = "chunk#"


@dataclass
class KbDoc:
    kbDocId: str
    sk: str
    docTitle: str
    sourceType: str
    filename: str
    contentType: str
    sizeBytes: int
    s3Key: str
    status: str
    uploadedBy: str
    createdAt: int
    updatedAt: int
    statusMessage: str = ""
    totalChunks: int = 0
    # Free-form label that groups related docs in the admin UI (e.g.
    # "NDBI Research", "Parent Orientation"). Empty string means ungrouped.
    collection: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class ChunkRecord:
    """Input record for a single chunk being written to the table.

    Matches the contract's chunk item schema. `embedding` is the raw 1024-dim
    float list returned by Titan Embed v2.
    """

    chunkIdx: int
    chunkText: str
    chunkTokens: int
    embedding: list[float]
    pageNumber: int | None = None
    sectionTitle: str | None = None


@dataclass
class Chunk:
    """Hydrated chunk row as read back from DDB."""

    kbDocId: str
    chunkIdx: int
    chunkText: str
    chunkTokens: int
    embedding: list[float]
    docTitle: str
    sourceType: str
    pageNumber: int | None = None
    sectionTitle: str | None = None
    createdAt: int = 0


class KbRepo:
    """DynamoDB repository for the `anna-chat-{env}-kb` table."""

    def __init__(self, *, kb_table: str, region: str) -> None:
        ddb = boto3.resource("dynamodb", region_name=region)
        self._table = ddb.Table(kb_table)

    # ---------- helpers ----------

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _new_kb_doc_id() -> str:
        return f"kd_{uuid.uuid4().hex[:16]}"

    @staticmethod
    def _chunk_sk(idx: int) -> str:
        return f"{CHUNK_SK_PREFIX}{idx:04d}"

    @staticmethod
    def _to_doc(item: dict[str, Any]) -> KbDoc:
        known = {
            "kbDocId",
            "sk",
            "docTitle",
            "sourceType",
            "filename",
            "contentType",
            "sizeBytes",
            "s3Key",
            "status",
            "statusMessage",
            "totalChunks",
            "uploadedBy",
            "collection",
            "tags",
            "createdAt",
            "updatedAt",
        }
        kwargs: dict[str, Any] = {k: item[k] for k in known if k in item}
        for numfield in ("sizeBytes", "totalChunks", "createdAt", "updatedAt"):
            if numfield in kwargs and kwargs[numfield] is not None:
                kwargs[numfield] = int(kwargs[numfield])
        if "tags" in kwargs and kwargs["tags"] is not None:
            # DDB string sets come back as a set; normalize to a list.
            kwargs["tags"] = list(kwargs["tags"])
        else:
            kwargs["tags"] = []
        return KbDoc(**kwargs)

    @staticmethod
    def _to_chunk(item: dict[str, Any]) -> Chunk:
        # DDB numbers round-trip as Decimal — convert floats and ints explicitly
        # so downstream math doesn't need to care about Decimal arithmetic.
        embedding_raw = item.get("embedding") or []
        embedding = [float(x) for x in embedding_raw]
        chunk_idx = int(item.get("chunkIdx", 0))
        chunk_tokens = int(item.get("chunkTokens", 0))
        created_at_raw = item.get("createdAt", 0)
        created_at = int(created_at_raw) if created_at_raw is not None else 0
        page_raw = item.get("pageNumber")
        page_number = int(page_raw) if page_raw is not None else None
        return Chunk(
            kbDocId=item["kbDocId"],
            chunkIdx=chunk_idx,
            chunkText=item.get("chunkText", ""),
            chunkTokens=chunk_tokens,
            embedding=embedding,
            docTitle=item.get("docTitle", ""),
            sourceType=item.get("sourceType", ""),
            pageNumber=page_number,
            sectionTitle=item.get("sectionTitle") or None,
            createdAt=created_at,
        )

    # ---------- META item CRUD ----------

    def create_doc(
        self,
        *,
        user_id: str,
        kb_doc_id: str,
        filename: str,
        content_type: str,
        size_bytes: int,
        s3_key: str,
        doc_title: str,
        source_type: str,
        collection: str = "",
    ) -> KbDoc:
        """Create the META item with status=uploading."""
        now_ms = self._now_ms()
        doc = KbDoc(
            kbDocId=kb_doc_id,
            sk=META_SK,
            docTitle=doc_title,
            sourceType=source_type,
            filename=filename,
            contentType=content_type,
            sizeBytes=size_bytes,
            s3Key=s3_key,
            status="uploading",
            uploadedBy=user_id,
            collection=collection,
            createdAt=now_ms,
            updatedAt=now_ms,
        )
        item = asdict(doc)
        # Drop `tags` when empty so we don't write an empty list — optional per
        # the contract (it's a string set for later filtering).
        if not item.get("tags"):
            item.pop("tags", None)
        # Drop `collection` when empty so a) we don't have a sparse index of
        # empty strings and b) legacy rows (pre-collection) don't look
        # different from brand-new ungrouped uploads.
        if not item.get("collection"):
            item.pop("collection", None)
        self._table.put_item(Item=item)
        return doc

    def get_doc(self, kb_doc_id: str) -> KbDoc | None:
        resp = self._table.get_item(Key={"kbDocId": kb_doc_id, "sk": META_SK})
        item = resp.get("Item")
        return self._to_doc(item) if item else None

    def list_docs(self) -> list[KbDoc]:
        """Scan for META rows (there is no per-user split — this table is shared)."""
        docs: list[KbDoc] = []
        scan_kwargs: dict[str, Any] = {
            "FilterExpression": Attr("sk").eq(META_SK),
        }
        while True:
            resp = self._table.scan(**scan_kwargs)
            for item in resp.get("Items", []):
                docs.append(self._to_doc(item))
            lek = resp.get("LastEvaluatedKey")
            if not lek:
                break
            scan_kwargs["ExclusiveStartKey"] = lek
        return docs

    def update_doc_status(
        self,
        *,
        kb_doc_id: str,
        status: str,
        status_message: str | None = None,
        total_chunks: int | None = None,
    ) -> None:
        expr_parts = ["#s = :s", "updatedAt = :u"]
        names = {"#s": "status"}
        values: dict[str, Any] = {":s": status, ":u": self._now_ms()}
        if status_message is not None:
            expr_parts.append("statusMessage = :m")
            values[":m"] = status_message
        if total_chunks is not None:
            expr_parts.append("totalChunks = :tc")
            values[":tc"] = int(total_chunks)
        expr = "SET " + ", ".join(expr_parts)
        self._table.update_item(
            Key={"kbDocId": kb_doc_id, "sk": META_SK},
            UpdateExpression=expr,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )

    # ---------- chunk writes / reads ----------

    def write_chunks(self, kb_doc_id: str, chunks: list[ChunkRecord]) -> None:
        """Batch-write chunk items. DDB `batch_writer` auto-chunks to 25/request.

        Important: DynamoDB (via boto3's resource interface) does NOT
        accept raw Python floats — it raises
        `TypeError("Float types are not supported. Use Decimal types
        instead.")` at serialization time. Titan Embed v2 returns
        1024-dim float vectors, so every embedding must be coerced to
        Decimal before writing. Going through `str()` first preserves
        the exact decimal representation and avoids the
        `Decimal(0.1) == 0.1000000000000000055...` binary-float gotcha.
        """
        if not chunks:
            return
        doc = self.get_doc(kb_doc_id)
        doc_title = doc.docTitle if doc else ""
        source_type = doc.sourceType if doc else ""
        now_ms = self._now_ms()
        with self._table.batch_writer() as batch:
            for chunk in chunks:
                embedding_decimal = [Decimal(str(x)) for x in chunk.embedding]
                item: dict[str, Any] = {
                    "kbDocId": kb_doc_id,
                    "sk": self._chunk_sk(chunk.chunkIdx),
                    "chunkIdx": int(chunk.chunkIdx),
                    "chunkText": chunk.chunkText,
                    "chunkTokens": int(chunk.chunkTokens),
                    "embedding": embedding_decimal,
                    "docTitle": doc_title,
                    "sourceType": source_type,
                    "createdAt": now_ms,
                }
                if chunk.pageNumber is not None:
                    item["pageNumber"] = int(chunk.pageNumber)
                if chunk.sectionTitle:
                    item["sectionTitle"] = chunk.sectionTitle
                batch.put_item(Item=item)

    def scan_all_chunks(self) -> list[Chunk]:
        """Scan the whole table for chunk rows. Paginates internally.

        Used by the retrieval path and cached at the Lambda container level
        (see anna_chat.kb_retrieve._CACHED_CHUNKS).
        """
        chunks: list[Chunk] = []
        scan_kwargs: dict[str, Any] = {
            "FilterExpression": Attr("sk").begins_with(CHUNK_SK_PREFIX),
        }
        while True:
            resp = self._table.scan(**scan_kwargs)
            for item in resp.get("Items", []):
                chunks.append(self._to_chunk(item))
            lek = resp.get("LastEvaluatedKey")
            if not lek:
                break
            scan_kwargs["ExclusiveStartKey"] = lek
        return chunks

    # ---------- delete ----------

    def delete_doc(self, kb_doc_id: str) -> None:
        """Delete META + all chunks for a document.

        Chunks are read first to collect their sort keys, then all items
        (META + chunks) are removed in a batched writer.
        """
        chunk_sks: list[str] = []
        query_kwargs: dict[str, Any] = {
            "KeyConditionExpression": (
                Key("kbDocId").eq(kb_doc_id)
                & Key("sk").begins_with(CHUNK_SK_PREFIX)
            ),
            "ProjectionExpression": "sk",
        }
        while True:
            resp = self._table.query(**query_kwargs)
            for item in resp.get("Items", []):
                chunk_sks.append(item["sk"])
            lek = resp.get("LastEvaluatedKey")
            if not lek:
                break
            query_kwargs["ExclusiveStartKey"] = lek

        with self._table.batch_writer() as batch:
            batch.delete_item(Key={"kbDocId": kb_doc_id, "sk": META_SK})
            for sk in chunk_sks:
                batch.delete_item(Key={"kbDocId": kb_doc_id, "sk": sk})

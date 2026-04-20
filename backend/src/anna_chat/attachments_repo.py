import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import boto3
from boto3.dynamodb.conditions import Attr, Key


@dataclass
class Attachment:
    userId: str
    attachmentId: str
    conversationId: str
    createdAt: int
    filename: str
    contentType: str
    sizeBytes: int
    s3Key: str
    status: str
    statusMessage: str = ""
    extractedText: str = ""
    extractedPreview: str = ""
    truncated: bool = False
    ttl: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


class AttachmentsRepo:
    """DynamoDB repository for the attachments table.

    Matches the contract in docs/ATTACHMENTS_CONTRACT.md. Uses the same style
    as anna_chat.ddb.Repository (resource API + Key conditions).
    """

    def __init__(
        self,
        *,
        attachments_table: str,
        region: str,
        message_ttl_days: int = 90,
        gsi_name: str = "conversationId-createdAt-index",
    ) -> None:
        ddb = boto3.resource("dynamodb", region_name=region)
        self._table = ddb.Table(attachments_table)
        self._ttl_seconds = message_ttl_days * 86400
        self._gsi_name = gsi_name

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _new_attachment_id() -> str:
        return f"att_{uuid.uuid4().hex[:16]}"

    @staticmethod
    def _to_attachment(item: dict[str, Any]) -> Attachment:
        known = {
            "userId",
            "attachmentId",
            "conversationId",
            "createdAt",
            "filename",
            "contentType",
            "sizeBytes",
            "s3Key",
            "status",
            "statusMessage",
            "extractedText",
            "extractedPreview",
            "truncated",
            "ttl",
        }
        kwargs = {k: item[k] for k in known if k in item}
        # Cast number types pulled back as Decimal.
        for numfield in ("createdAt", "sizeBytes", "ttl"):
            if numfield in kwargs:
                kwargs[numfield] = int(kwargs[numfield])
        if "truncated" in kwargs:
            kwargs["truncated"] = bool(kwargs["truncated"])
        return Attachment(**kwargs)

    def create_attachment(
        self,
        *,
        user_id: str,
        conversation_id: str,
        filename: str,
        content_type: str,
        size_bytes: int,
        s3_key: str,
        status: str = "uploading",
        attachment_id: str | None = None,
    ) -> Attachment:
        now_ms = self._now_ms()
        att = Attachment(
            userId=user_id,
            attachmentId=attachment_id or self._new_attachment_id(),
            conversationId=conversation_id,
            createdAt=now_ms,
            filename=filename,
            contentType=content_type,
            sizeBytes=size_bytes,
            s3Key=s3_key,
            status=status,
            ttl=int(time.time()) + self._ttl_seconds,
        )
        item = asdict(att)
        item.pop("extras", None)
        self._table.put_item(Item=item)
        return att

    def get_attachment(
        self, *, user_id: str, attachment_id: str
    ) -> Attachment | None:
        resp = self._table.get_item(
            Key={"userId": user_id, "attachmentId": attachment_id}
        )
        item = resp.get("Item")
        return self._to_attachment(item) if item else None

    def list_for_conversation(
        self,
        *,
        conversation_id: str,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Attachment]:
        query_kwargs: dict[str, Any] = {
            "IndexName": self._gsi_name,
            "KeyConditionExpression": Key("conversationId").eq(conversation_id),
            "Limit": limit,
            "ScanIndexForward": True,
        }
        if status:
            query_kwargs["FilterExpression"] = Attr("status").eq(status)
        resp = self._table.query(**query_kwargs)
        return [self._to_attachment(item) for item in resp.get("Items", [])]

    def update_status(
        self,
        *,
        user_id: str,
        attachment_id: str,
        status: str,
        status_message: str | None = None,
    ) -> None:
        expr = "SET #s = :s"
        names = {"#s": "status"}
        values: dict[str, Any] = {":s": status}
        if status_message is not None:
            expr += ", statusMessage = :m"
            values[":m"] = status_message
        self._table.update_item(
            Key={"userId": user_id, "attachmentId": attachment_id},
            UpdateExpression=expr,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )

    def set_extraction_result(
        self,
        *,
        user_id: str,
        attachment_id: str,
        extracted_text: str,
        truncated: bool,
        preview_chars: int = 300,
    ) -> None:
        preview = extracted_text[:preview_chars]
        self._table.update_item(
            Key={"userId": user_id, "attachmentId": attachment_id},
            UpdateExpression=(
                "SET #s = :s, extractedText = :t, extractedPreview = :p, "
                "truncated = :tr, statusMessage = :m"
            ),
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":s": "ready",
                ":t": extracted_text,
                ":p": preview,
                ":tr": truncated,
                ":m": "",
            },
        )

    def delete(self, *, user_id: str, attachment_id: str) -> None:
        self._table.delete_item(
            Key={"userId": user_id, "attachmentId": attachment_id}
        )

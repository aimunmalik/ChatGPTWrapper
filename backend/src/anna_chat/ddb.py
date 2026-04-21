import time
import uuid
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key


def _floats_to_decimal(value: Any) -> Any:
    """Recursively convert floats → Decimal so boto3's DDB resource can serialize.

    boto3's resource-level serializer rejects raw Python floats with
    `TypeError("Float types are not supported. Use Decimal types instead.")`.
    Anywhere user-generated or LLM-generated content can introduce a float
    (e.g. similarity scores on the assistant message's `sources`), we have
    to convert before `put_item`.

    Goes through `str()` on the way to Decimal to preserve Titan / Bedrock's
    exact reported precision and dodge the `Decimal(0.1) == 0.1000...55`
    binary-float gotcha.
    """
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _floats_to_decimal(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_floats_to_decimal(v) for v in value]
    return value


@dataclass
class Conversation:
    userId: str
    conversationId: str
    title: str
    createdAt: int
    updatedAt: int
    model: str


@dataclass
class Message:
    conversationId: str
    sortKey: str
    userId: str
    role: str
    content: str
    messageId: str
    inputTokens: int = 0
    outputTokens: int = 0
    model: str = ""
    ttl: int = 0
    sources: list[dict[str, Any]] = field(default_factory=list)


class Repository:
    def __init__(
        self,
        *,
        conversations_table: str,
        messages_table: str,
        region: str,
        message_ttl_days: int,
    ) -> None:
        ddb = boto3.resource("dynamodb", region_name=region)
        self._conversations = ddb.Table(conversations_table)
        self._messages = ddb.Table(messages_table)
        self._ttl_seconds = message_ttl_days * 86400

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _sort_key(timestamp_ms: int, message_id: str) -> str:
        return f"{timestamp_ms:013d}#{message_id}"

    def create_conversation(self, *, user_id: str, title: str, model: str) -> Conversation:
        now = self._now_ms()
        conv = Conversation(
            userId=user_id,
            conversationId=f"c_{uuid.uuid4().hex[:16]}",
            title=title[:120],
            createdAt=now,
            updatedAt=now,
            model=model,
        )
        self._conversations.put_item(Item=asdict(conv))
        return conv

    def get_conversation(self, *, user_id: str, conversation_id: str) -> Conversation | None:
        resp = self._conversations.get_item(
            Key={"userId": user_id, "conversationId": conversation_id}
        )
        item = resp.get("Item")
        return Conversation(**item) if item else None

    def list_conversations(self, *, user_id: str, limit: int = 50) -> list[Conversation]:
        resp = self._conversations.query(
            KeyConditionExpression=Key("userId").eq(user_id),
            Limit=limit,
            ScanIndexForward=False,
        )
        return [Conversation(**item) for item in resp.get("Items", [])]

    def delete_conversation(self, *, user_id: str, conversation_id: str) -> None:
        self._conversations.delete_item(
            Key={"userId": user_id, "conversationId": conversation_id}
        )
        messages = self.list_messages(conversation_id=conversation_id)
        with self._messages.batch_writer() as batch:
            for msg in messages:
                batch.delete_item(
                    Key={"conversationId": msg.conversationId, "sortKey": msg.sortKey}
                )

    def touch_conversation(self, *, user_id: str, conversation_id: str) -> None:
        self._conversations.update_item(
            Key={"userId": user_id, "conversationId": conversation_id},
            UpdateExpression="SET updatedAt = :now",
            ExpressionAttributeValues={":now": self._now_ms()},
        )

    def append_message(
        self,
        *,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
        sources: list[dict[str, Any]] | None = None,
    ) -> Message:
        now = self._now_ms()
        message_id = f"m_{uuid.uuid4().hex[:16]}"
        msg = Message(
            conversationId=conversation_id,
            sortKey=self._sort_key(now, message_id),
            userId=user_id,
            role=role,
            content=content,
            messageId=message_id,
            inputTokens=input_tokens,
            outputTokens=output_tokens,
            model=model,
            ttl=int(time.time()) + self._ttl_seconds,
            sources=list(sources) if sources else [],
        )
        # Sources carry a float `score` from the KB retriever. If we passed
        # that straight to DDB, the resource serializer would TypeError on
        # the first float it saw — same bug that used to break every KB
        # write before we converted embedding vectors. Sanitize the whole
        # item once at the boundary.
        self._messages.put_item(Item=_floats_to_decimal(asdict(msg)))
        return msg

    def list_messages(self, *, conversation_id: str, limit: int = 200) -> list[Message]:
        resp = self._messages.query(
            KeyConditionExpression=Key("conversationId").eq(conversation_id),
            Limit=limit,
            ScanIndexForward=True,
        )
        messages: list[Message] = []
        for item in resp.get("Items", []):
            # Backfill sources for older rows that were written before the
            # field existed. Defaulting at read time keeps the dataclass
            # contract consistent regardless of when the row was created.
            if "sources" not in item:
                item["sources"] = []
            messages.append(Message(**item))
        return messages

    def recent_turns_for_model(
        self, *, conversation_id: str, max_turns: int = 20
    ) -> list[dict[str, Any]]:
        msgs = self.list_messages(conversation_id=conversation_id, limit=max_turns * 2)
        return [{"role": m.role, "content": m.content} for m in msgs if m.role in {"user", "assistant"}]

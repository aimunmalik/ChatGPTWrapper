import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key


@dataclass
class Prompt:
    userId: str
    promptId: str
    title: str
    body: str
    createdAt: int
    updatedAt: int


class PromptsRepo:
    """DynamoDB repository for the prompts table.

    Matches the contract in docs/PROMPTS_CONTRACT.md. Uses the same style as
    anna_chat.attachments_repo.AttachmentsRepo (resource API + Key conditions).
    """

    def __init__(
        self,
        *,
        prompts_table: str,
        region: str,
    ) -> None:
        ddb = boto3.resource("dynamodb", region_name=region)
        self._table = ddb.Table(prompts_table)

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _new_prompt_id() -> str:
        return f"p_{uuid.uuid4().hex[:16]}"

    @staticmethod
    def _to_prompt(item: dict[str, Any]) -> Prompt:
        known = {"userId", "promptId", "title", "body", "createdAt", "updatedAt"}
        kwargs = {k: item[k] for k in known if k in item}
        for numfield in ("createdAt", "updatedAt"):
            if numfield in kwargs:
                kwargs[numfield] = int(kwargs[numfield])
        return Prompt(**kwargs)

    def create(
        self,
        *,
        user_id: str,
        title: str,
        body: str,
        prompt_id: str | None = None,
    ) -> Prompt:
        now_ms = self._now_ms()
        prompt = Prompt(
            userId=user_id,
            promptId=prompt_id or self._new_prompt_id(),
            title=title,
            body=body,
            createdAt=now_ms,
            updatedAt=now_ms,
        )
        self._table.put_item(Item=asdict(prompt))
        return prompt

    def list_for_user(self, *, user_id: str) -> list[Prompt]:
        resp = self._table.query(
            KeyConditionExpression=Key("userId").eq(user_id),
        )
        return [self._to_prompt(item) for item in resp.get("Items", [])]

    def get(self, *, user_id: str, prompt_id: str) -> Prompt | None:
        resp = self._table.get_item(
            Key={"userId": user_id, "promptId": prompt_id}
        )
        item = resp.get("Item")
        return self._to_prompt(item) if item else None

    def update(
        self,
        *,
        user_id: str,
        prompt_id: str,
        title: str,
        body: str,
    ) -> Prompt | None:
        """Full-replace update. Returns the updated prompt, or None if not
        owned by the user / not found.

        Uses a condition expression on the partition+sort key pair to enforce
        ownership atomically.
        """
        now_ms = self._now_ms()
        try:
            resp = self._table.update_item(
                Key={"userId": user_id, "promptId": prompt_id},
                UpdateExpression=(
                    "SET title = :t, body = :b, updatedAt = :u"
                ),
                ConditionExpression=(
                    "attribute_exists(userId) AND attribute_exists(promptId)"
                ),
                ExpressionAttributeValues={
                    ":t": title,
                    ":b": body,
                    ":u": now_ms,
                },
                ReturnValues="ALL_NEW",
            )
        except self._table.meta.client.exceptions.ConditionalCheckFailedException:
            return None
        item = resp.get("Attributes")
        return self._to_prompt(item) if item else None

    def delete(self, *, user_id: str, prompt_id: str) -> bool:
        """Delete a prompt. Returns True if a row existed, False otherwise."""
        resp = self._table.delete_item(
            Key={"userId": user_id, "promptId": prompt_id},
            ReturnValues="ALL_OLD",
        )
        return bool(resp.get("Attributes"))

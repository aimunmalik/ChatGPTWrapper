import json
from dataclasses import dataclass
from typing import Any

import boto3


@dataclass(frozen=True)
class BedrockResponse:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str


class BedrockClient:
    def __init__(self, *, region: str, model_id: str) -> None:
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def invoke(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        model_id: str | None = None,
    ) -> BedrockResponse:
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            body["system"] = system

        resp = self._client.invoke_model(
            modelId=model_id or self._model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        payload = json.loads(resp["body"].read())
        text = "".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        )
        usage = payload.get("usage", {})
        return BedrockResponse(
            text=text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            stop_reason=payload.get("stop_reason", "unknown"),
        )

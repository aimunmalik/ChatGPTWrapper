import json
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.config import Config


@dataclass(frozen=True)
class BedrockResponse:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str


# Default timeouts are tuned for the SYNCHRONOUS chat handler, which must
# fit inside API Gateway HTTP API's hard 30s integration cap. With default
# boto retries (3 attempts + exponential backoff) one ThrottlingException
# burst can eat the whole window and 503 the client before Bedrock finishes.
# 25s read + 1 retry covers a transient throttle without blowing past API GW.
SYNC_READ_TIMEOUT = 25
# Async / background callers (e.g. the translate worker) have a 15-min
# Lambda budget — they don't need to fail fast. A long-form translation
# chunk can legitimately take 60s+ when generating 3000+ output tokens.
ASYNC_READ_TIMEOUT = 180


class BedrockClient:
    """Thin Bedrock InvokeModel wrapper with timeout/retry tuning per caller.

    `read_timeout` defaults to the SYNC value (25s) — appropriate for the
    chat handler. Background workers should pass a higher value:

        BedrockClient(region=..., model_id=..., read_timeout=ASYNC_READ_TIMEOUT)
    """

    def __init__(
        self,
        *,
        region: str,
        model_id: str,
        read_timeout: int = SYNC_READ_TIMEOUT,
    ) -> None:
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                connect_timeout=5,
                read_timeout=read_timeout,
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        )
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

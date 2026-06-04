import json
from collections.abc import Iterator
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
# Streaming chat: bytes arrive incrementally, so the read timeout governs the
# GAP between chunks (time-to-first-byte / inter-token), not total generation
# time. 120s is generous — if Bedrock stalls mid-stream that long, something is
# wrong. The streaming client also uses max_attempts=1: a partially-consumed
# stream must never be retried (it would replay already-emitted tokens).
STREAM_READ_TIMEOUT = 120


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
        max_attempts: int = 2,
    ) -> None:
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                connect_timeout=5,
                read_timeout=read_timeout,
                retries={"max_attempts": max_attempts, "mode": "standard"},
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

    def invoke_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        model_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream a Bedrock completion as incremental events.

        Yields dicts in order:
          {"type": "delta", "text": "..."}    incremental output text (0..N)
          {"type": "done", "inputTokens": int, "outputTokens": int,
           "stopReason": str}                 terminal, emitted exactly once

        Uses invoke_model_with_response_stream. The caller accumulates delta
        text for persistence. Construct this client with max_attempts=1 so a
        partially-consumed stream is never retried (which would replay tokens).
        """
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            body["system"] = system

        resp = self._client.invoke_model_with_response_stream(
            modelId=model_id or self._model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        input_tokens = 0
        output_tokens = 0
        stop_reason = "unknown"
        for event in resp["body"]:
            chunk = event.get("chunk")
            if not chunk:
                continue
            payload = json.loads(chunk["bytes"].decode("utf-8"))
            ptype = payload.get("type")
            if ptype == "message_start":
                usage = payload.get("message", {}).get("usage", {})
                input_tokens = usage.get("input_tokens", input_tokens)
            elif ptype == "content_block_delta":
                delta = payload.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        yield {"type": "delta", "text": text}
            elif ptype == "message_delta":
                usage = payload.get("usage", {})
                output_tokens = usage.get("output_tokens", output_tokens)
                stop = payload.get("delta", {}).get("stop_reason")
                if stop:
                    stop_reason = stop
        yield {
            "type": "done",
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "stopReason": stop_reason,
        }

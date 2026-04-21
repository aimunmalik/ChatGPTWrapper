"""Thin Bedrock Runtime wrapper for Amazon Titan Text Embeddings v2.

Returns a 1024-dim normalized vector per input string. See the KB contract
(docs/KB_CONTRACT.md) — Titan v2 is HIPAA-eligible via the AWS BAA, supports
cosine similarity directly on normalized output, and needs the user to enable
the model in the Bedrock console (same "Model access" flow as Claude).

Request body for Titan Embed v2:
    {"inputText": "...", "dimensions": 1024, "normalize": true}

Response body:
    {"embedding": [float, ...], "inputTextTokenCount": int}
"""

from __future__ import annotations

import json
from typing import Any

import boto3

DEFAULT_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_DIMENSIONS = 1024


class EmbeddingsClient:
    """Wrapper around the Bedrock Runtime InvokeModel call for Titan Embed v2."""

    def __init__(
        self,
        region: str,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        dimensions: int = DEFAULT_DIMENSIONS,
        client: Any | None = None,
    ) -> None:
        self._client = client or boto3.client("bedrock-runtime", region_name=region)
        self._model_id = model_id
        self._dimensions = dimensions

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        """Return a 1024-dim unit-normalized embedding for `text`.

        The caller should truncate or chunk long inputs — Titan v2's input
        limit is ~8192 tokens. Empty strings are rejected by Bedrock with a
        validation error; we short-circuit to return a zero vector so callers
        don't pay for a round-trip on empty paragraphs.
        """
        if not text or not text.strip():
            return [0.0] * self._dimensions

        body = {
            "inputText": text,
            "dimensions": self._dimensions,
            "normalize": True,
        }
        resp = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(resp["body"].read())
        return [float(v) for v in payload.get("embedding", [])]

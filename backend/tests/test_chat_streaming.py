"""Tests for the streaming-chat building blocks (see docs/STREAMING_CONTRACT.md):

  - BedrockClient.invoke_stream() event parsing
  - Message.status default + the messages-repo streaming methods
  - recent_turns_for_model excluding in-flight `streaming` placeholders

DynamoDB calls are exercised with botocore Stubber (same approach as
test_jobs_repo); the Bedrock event stream is faked with a plain list of chunk
dicts, which is exactly what boto3's EventStream yields when iterated.
"""

import json
from unittest.mock import MagicMock

from botocore.stub import Stubber

from anna_chat.bedrock_client import BedrockClient
from anna_chat.ddb import Message, Repository

MSG_TABLE = "t-msg"
CONV_TABLE = "t-conv"


def _make_repo() -> Repository:
    return Repository(
        conversations_table=CONV_TABLE,
        messages_table=MSG_TABLE,
        region="us-east-1",
        message_ttl_days=90,
    )


def _evt(payload: dict) -> dict:
    """Shape a payload the way boto3's response stream yields it."""
    return {"chunk": {"bytes": json.dumps(payload).encode("utf-8")}}


# --------------------------------------------------------------------------- #
# BedrockClient.invoke_stream
# --------------------------------------------------------------------------- #
def test_invoke_stream_yields_deltas_then_done():
    client = BedrockClient(
        region="us-east-1", model_id="m", read_timeout=120, max_attempts=1
    )
    fake = MagicMock()
    fake.invoke_model_with_response_stream.return_value = {
        "body": [
            _evt({"type": "message_start", "message": {"usage": {"input_tokens": 42}}}),
            _evt(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hello"},
                }
            ),
            _evt(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": " world"},
                }
            ),
            _evt(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 7},
                }
            ),
            _evt({"type": "message_stop"}),
        ]
    }
    client._client = fake  # noqa: SLF001 — test-only injection

    events = list(
        client.invoke_stream(
            messages=[{"role": "user", "content": "hi"}], system="sys", model_id="m"
        )
    )

    deltas = [e["text"] for e in events if e["type"] == "delta"]
    done = [e for e in events if e["type"] == "done"]
    assert deltas == ["Hello", " world"]
    assert len(done) == 1
    assert done[0]["inputTokens"] == 42
    assert done[0]["outputTokens"] == 7
    assert done[0]["stopReason"] == "end_turn"

    # The request carried system + messages through to Bedrock.
    _, kwargs = fake.invoke_model_with_response_stream.call_args
    body = json.loads(kwargs["body"])
    assert body["system"] == "sys"
    assert body["messages"][0]["content"] == "hi"


def test_invoke_stream_emits_done_even_with_no_text():
    client = BedrockClient(region="us-east-1", model_id="m", max_attempts=1)
    fake = MagicMock()
    fake.invoke_model_with_response_stream.return_value = {
        "body": [
            _evt({"type": "message_start", "message": {"usage": {"input_tokens": 5}}}),
            _evt({"type": "message_stop"}),
        ]
    }
    client._client = fake  # noqa: SLF001
    events = list(client.invoke_stream(messages=[{"role": "user", "content": "x"}]))
    assert events == [
        {"type": "done", "inputTokens": 5, "outputTokens": 0, "stopReason": "unknown"}
    ]


# --------------------------------------------------------------------------- #
# Message.status
# --------------------------------------------------------------------------- #
def test_message_defaults_to_complete_status():
    m = Message(
        conversationId="c",
        sortKey="s",
        userId="u",
        role="assistant",
        content="hi",
        messageId="m1",
    )
    assert m.status == "complete"


# --------------------------------------------------------------------------- #
# Messages repo streaming methods
# --------------------------------------------------------------------------- #
def test_create_streaming_message_writes_streaming_row():
    repo = _make_repo()
    stub = Stubber(repo._messages.meta.client)  # noqa: SLF001
    stub.add_response("put_item", {})
    stub.activate()
    try:
        msg = repo.create_streaming_message(
            conversation_id="c1", user_id="u1", model="us.anthropic.claude-sonnet-4-6"
        )
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()

    assert msg.role == "assistant"
    assert msg.status == "streaming"
    assert msg.content == ""
    assert msg.messageId.startswith("m_")
    assert msg.sortKey.endswith(f"#{msg.messageId}")


def test_get_message_returns_none_when_absent():
    repo = _make_repo()
    stub = Stubber(repo._messages.meta.client)  # noqa: SLF001
    stub.add_response("get_item", {})  # no "Item"
    stub.activate()
    try:
        assert repo.get_message(conversation_id="c1", sort_key="sk") is None
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()


def test_get_message_hydrates_streaming_row():
    repo = _make_repo()
    stub = Stubber(repo._messages.meta.client)  # noqa: SLF001
    stub.add_response(
        "get_item",
        {
            "Item": {
                "conversationId": {"S": "c1"},
                "sortKey": {"S": "0000000000002#m_a"},
                "userId": {"S": "u1"},
                "role": {"S": "assistant"},
                "content": {"S": "partial..."},
                "messageId": {"S": "m_a"},
                "status": {"S": "streaming"},
            }
        },
    )
    stub.activate()
    try:
        msg = repo.get_message(conversation_id="c1", sort_key="0000000000002#m_a")
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()

    assert msg is not None
    assert msg.status == "streaming"
    assert msg.content == "partial..."
    assert msg.sources == []  # defaulted for rows that predate the field


def test_recent_turns_excludes_streaming_placeholder():
    repo = _make_repo()
    stub = Stubber(repo._messages.meta.client)  # noqa: SLF001
    stub.add_response(
        "query",
        {
            "Items": [
                {
                    "conversationId": {"S": "c1"},
                    "sortKey": {"S": "0000000000001#m_u"},
                    "userId": {"S": "u1"},
                    "role": {"S": "user"},
                    "content": {"S": "hi"},
                    "messageId": {"S": "m_u"},
                    "status": {"S": "complete"},
                },
                {
                    "conversationId": {"S": "c1"},
                    "sortKey": {"S": "0000000000002#m_a"},
                    "userId": {"S": "u1"},
                    "role": {"S": "assistant"},
                    "content": {"S": ""},
                    "messageId": {"S": "m_a"},
                    "status": {"S": "streaming"},
                },
            ]
        },
    )
    stub.activate()
    try:
        turns = repo.recent_turns_for_model(conversation_id="c1")
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()

    # The in-flight streaming placeholder must NOT leak into model history.
    assert turns == [{"role": "user", "content": "hi"}]

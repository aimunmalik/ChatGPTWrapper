from decimal import Decimal
from unittest.mock import patch

from botocore.stub import Stubber

from anna_chat.attachments_repo import Attachment, AttachmentsRepo

TABLE = "anna-chat-test-attachments"


def _make_repo() -> tuple[AttachmentsRepo, Stubber]:
    repo = AttachmentsRepo(
        attachments_table=TABLE,
        region="us-east-1",
        message_ttl_days=90,
    )
    client = repo._table.meta.client  # noqa: SLF001 — test-only access
    stub = Stubber(client)
    stub.activate()
    return repo, stub


def test_create_attachment_puts_expected_item():
    repo, stub = _make_repo()
    stub.add_response(
        "put_item",
        {},
        expected_params={
            "TableName": TABLE,
            # We don't assert the full item because attachmentId/timestamps
            # are generated. Only require TableName + that Item is a dict.
            "Item": {
                "userId": "u1",
                "attachmentId": _anything_starting_with("att_"),
                "conversationId": "c1",
                "createdAt": _any_int(),
                "filename": "plan.pdf",
                "contentType": "application/pdf",
                "sizeBytes": 1234,
                "s3Key": "attachments/u1/att_fixed/plan.pdf",
                "status": "uploading",
                "statusMessage": "",
                "extractedText": "",
                "extractedPreview": "",
                "truncated": False,
                "ttl": _any_int(),
            },
        },
    )
    try:
        att = repo.create_attachment(
            user_id="u1",
            conversation_id="c1",
            filename="plan.pdf",
            content_type="application/pdf",
            size_bytes=1234,
            s3_key="attachments/u1/att_fixed/plan.pdf",
            attachment_id="att_fixed",
        )
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()

    assert isinstance(att, Attachment)
    assert att.attachmentId == "att_fixed"
    assert att.status == "uploading"
    assert att.sizeBytes == 1234


def test_get_attachment_returns_dataclass_with_decimals_cast():
    repo, stub = _make_repo()
    stub.add_response(
        "get_item",
        {
            "Item": {
                "userId": {"S": "u1"},
                "attachmentId": {"S": "att_abc"},
                "conversationId": {"S": "c1"},
                "createdAt": {"N": "1700000000000"},
                "filename": {"S": "f.pdf"},
                "contentType": {"S": "application/pdf"},
                "sizeBytes": {"N": "100"},
                "s3Key": {"S": "attachments/u1/att_abc/f.pdf"},
                "status": {"S": "ready"},
                "statusMessage": {"S": ""},
                "extractedText": {"S": "hello"},
                "extractedPreview": {"S": "hello"},
                "truncated": {"BOOL": False},
                "ttl": {"N": "1800000000"},
            }
        },
        expected_params={
            "TableName": TABLE,
            "Key": {"userId": "u1", "attachmentId": "att_abc"},
        },
    )
    try:
        att = repo.get_attachment(user_id="u1", attachment_id="att_abc")
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()

    assert att is not None
    assert att.attachmentId == "att_abc"
    assert isinstance(att.sizeBytes, int)
    assert att.sizeBytes == 100
    assert att.createdAt == 1700000000000
    assert att.status == "ready"
    assert att.truncated is False


def test_get_attachment_returns_none_when_missing():
    repo, stub = _make_repo()
    stub.add_response(
        "get_item",
        {},
        expected_params={
            "TableName": TABLE,
            "Key": {"userId": "u1", "attachmentId": "missing"},
        },
    )
    try:
        att = repo.get_attachment(user_id="u1", attachment_id="missing")
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()
    assert att is None


def test_list_for_conversation_queries_gsi():
    repo, stub = _make_repo()
    # The boto3 resource layer passes ConditionBase objects to the low-level
    # client, and the Stubber matches raw params — so we skip expected_params
    # for Query and instead assert on the observed call + returned items.
    stub.add_response(
        "query",
        {
            "Items": [
                {
                    "userId": {"S": "u1"},
                    "attachmentId": {"S": "att_1"},
                    "conversationId": {"S": "c1"},
                    "createdAt": {"N": "1700000000000"},
                    "filename": {"S": "a.pdf"},
                    "contentType": {"S": "application/pdf"},
                    "sizeBytes": {"N": "10"},
                    "s3Key": {"S": "attachments/u1/att_1/a.pdf"},
                    "status": {"S": "ready"},
                    "statusMessage": {"S": ""},
                    "extractedText": {"S": "x"},
                    "extractedPreview": {"S": "x"},
                    "truncated": {"BOOL": False},
                    "ttl": {"N": "0"},
                },
            ]
        },
    )
    try:
        items = repo.list_for_conversation(conversation_id="c1")
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()
    assert len(items) == 1
    assert items[0].attachmentId == "att_1"
    assert items[0].status == "ready"
    assert items[0].sizeBytes == 10


def test_list_for_conversation_with_status_filter():
    repo, stub = _make_repo()
    captured: list[dict] = []

    with patch.object(
        repo._table,
        "query",
        wraps=lambda **kwargs: (captured.append(kwargs) or {"Items": []}),
    ):
        items = repo.list_for_conversation(
            conversation_id="c1", status="ready"
        )
    stub.deactivate()
    assert items == []
    assert len(captured) == 1
    assert captured[0]["IndexName"] == "conversationId-createdAt-index"
    assert "FilterExpression" in captured[0]
    # KeyConditionExpression and FilterExpression are ConditionBase objects;
    # inspect their values to confirm correct attributes were matched.
    key_expr = captured[0]["KeyConditionExpression"]
    filter_expr = captured[0]["FilterExpression"]
    assert key_expr.get_expression()["values"][0].name == "conversationId"
    assert filter_expr.get_expression()["values"][0].name == "status"
    assert key_expr.get_expression()["values"][1] == "c1"
    assert filter_expr.get_expression()["values"][1] == "ready"


def test_update_status_sets_status_and_message():
    repo, stub = _make_repo()
    stub.add_response(
        "update_item",
        {},
        expected_params={
            "TableName": TABLE,
            "Key": {"userId": "u1", "attachmentId": "att_1"},
            "UpdateExpression": "SET #s = :s, statusMessage = :m",
            "ExpressionAttributeNames": {"#s": "status"},
            "ExpressionAttributeValues": {":s": "error", ":m": "boom"},
        },
    )
    try:
        repo.update_status(
            user_id="u1",
            attachment_id="att_1",
            status="error",
            status_message="boom",
        )
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()


def test_set_extraction_result_writes_text_preview_and_flags():
    repo, stub = _make_repo()
    long_text = "Z" * 500
    preview = long_text[:300]
    stub.add_response(
        "update_item",
        {},
        expected_params={
            "TableName": TABLE,
            "Key": {"userId": "u1", "attachmentId": "att_1"},
            "UpdateExpression": (
                "SET #s = :s, extractedText = :t, extractedPreview = :p, "
                "truncated = :tr, statusMessage = :m"
            ),
            "ExpressionAttributeNames": {"#s": "status"},
            "ExpressionAttributeValues": {
                ":s": "ready",
                ":t": long_text,
                ":p": preview,
                ":tr": True,
                ":m": "",
            },
        },
    )
    try:
        repo.set_extraction_result(
            user_id="u1",
            attachment_id="att_1",
            extracted_text=long_text,
            truncated=True,
        )
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()


def test_delete_deletes_row():
    repo, stub = _make_repo()
    stub.add_response(
        "delete_item",
        {},
        expected_params={
            "TableName": TABLE,
            "Key": {"userId": "u1", "attachmentId": "att_1"},
        },
    )
    try:
        repo.delete(user_id="u1", attachment_id="att_1")
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()


# ---------- helpers ----------


class _AnyInt:
    def __eq__(self, other):
        return isinstance(other, int | Decimal)

    def __repr__(self) -> str:  # pragma: no cover
        return "_AnyInt()"


class _StartsWith:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def __eq__(self, other) -> bool:
        return isinstance(other, str) and other.startswith(self.prefix)

    def __repr__(self) -> str:  # pragma: no cover
        return f"_StartsWith({self.prefix!r})"


def _any_int() -> _AnyInt:
    return _AnyInt()


def _anything_starting_with(prefix: str) -> _StartsWith:
    return _StartsWith(prefix)

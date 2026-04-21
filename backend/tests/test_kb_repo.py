from decimal import Decimal
from unittest.mock import patch

from botocore.stub import Stubber

from anna_chat.kb_repo import Chunk, ChunkRecord, KbDoc, KbRepo

TABLE = "anna-chat-test-kb"


def _make_repo() -> tuple[KbRepo, Stubber]:
    repo = KbRepo(kb_table=TABLE, region="us-east-1")
    client = repo._table.meta.client  # noqa: SLF001 — test-only access
    stub = Stubber(client)
    stub.activate()
    return repo, stub


def test_new_kb_doc_id_shape():
    kb_id = KbRepo._new_kb_doc_id()
    assert kb_id.startswith("kd_")
    hex_part = kb_id[3:]
    assert len(hex_part) == 16
    int(hex_part, 16)


def test_chunk_sk_zero_padded_to_four_digits():
    assert KbRepo._chunk_sk(0) == "chunk#0000"
    assert KbRepo._chunk_sk(7) == "chunk#0007"
    assert KbRepo._chunk_sk(123) == "chunk#0123"
    assert KbRepo._chunk_sk(9999) == "chunk#9999"


def test_create_doc_writes_meta_item_with_uploading_status():
    repo, stub = _make_repo()
    stub.add_response(
        "put_item",
        {},
        expected_params={
            "TableName": TABLE,
            "Item": {
                "kbDocId": "kd_fixed0123456789",
                "sk": "META",
                "docTitle": "ANNA Training",
                "sourceType": "training",
                "filename": "t.pdf",
                "contentType": "application/pdf",
                "sizeBytes": 4242,
                "s3Key": "kb/kd_fixed0123456789/t.pdf",
                "status": "uploading",
                "statusMessage": "",
                "totalChunks": 0,
                "uploadedBy": "u_admin",
                "createdAt": _any_int(),
                "updatedAt": _any_int(),
            },
        },
    )
    try:
        doc = repo.create_doc(
            user_id="u_admin",
            kb_doc_id="kd_fixed0123456789",
            filename="t.pdf",
            content_type="application/pdf",
            size_bytes=4242,
            s3_key="kb/kd_fixed0123456789/t.pdf",
            doc_title="ANNA Training",
            source_type="training",
        )
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()
    assert isinstance(doc, KbDoc)
    assert doc.status == "uploading"
    assert doc.sk == "META"
    assert doc.totalChunks == 0


def test_get_doc_returns_dataclass_with_ints_cast():
    repo, stub = _make_repo()
    stub.add_response(
        "get_item",
        {
            "Item": {
                "kbDocId": {"S": "kd_abc"},
                "sk": {"S": "META"},
                "docTitle": {"S": "Doc"},
                "sourceType": {"S": "research"},
                "filename": {"S": "d.pdf"},
                "contentType": {"S": "application/pdf"},
                "sizeBytes": {"N": "1000"},
                "s3Key": {"S": "kb/kd_abc/d.pdf"},
                "status": {"S": "ready"},
                "statusMessage": {"S": ""},
                "totalChunks": {"N": "12"},
                "uploadedBy": {"S": "u_1"},
                "createdAt": {"N": "1700000000000"},
                "updatedAt": {"N": "1700000001000"},
            }
        },
        expected_params={
            "TableName": TABLE,
            "Key": {"kbDocId": "kd_abc", "sk": "META"},
        },
    )
    try:
        doc = repo.get_doc("kd_abc")
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()
    assert doc is not None
    assert doc.kbDocId == "kd_abc"
    assert doc.status == "ready"
    assert isinstance(doc.sizeBytes, int)
    assert doc.sizeBytes == 1000
    assert doc.totalChunks == 12


def test_get_doc_returns_none_when_missing():
    repo, stub = _make_repo()
    stub.add_response(
        "get_item",
        {},
        expected_params={
            "TableName": TABLE,
            "Key": {"kbDocId": "kd_missing", "sk": "META"},
        },
    )
    try:
        doc = repo.get_doc("kd_missing")
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()
    assert doc is None


def test_list_docs_scans_for_meta_filter():
    repo, stub = _make_repo()
    captured: list[dict] = []
    with patch.object(
        repo._table,
        "scan",
        wraps=lambda **kwargs: (captured.append(kwargs) or {"Items": []}),
    ):
        items = repo.list_docs()
    stub.deactivate()
    assert items == []
    assert len(captured) == 1
    # FilterExpression should filter on sk == META.
    filter_expr = captured[0]["FilterExpression"]
    assert filter_expr.get_expression()["values"][0].name == "sk"
    assert filter_expr.get_expression()["values"][1] == "META"


def test_update_doc_status_sets_status_and_total_chunks():
    repo, stub = _make_repo()
    stub.add_response(
        "update_item",
        {},
        expected_params={
            "TableName": TABLE,
            "Key": {"kbDocId": "kd_1", "sk": "META"},
            "UpdateExpression": (
                "SET #s = :s, updatedAt = :u, statusMessage = :m, totalChunks = :tc"
            ),
            "ExpressionAttributeNames": {"#s": "status"},
            "ExpressionAttributeValues": {
                ":s": "ready",
                ":u": _any_int(),
                ":m": "",
                ":tc": 42,
            },
        },
    )
    try:
        repo.update_doc_status(
            kb_doc_id="kd_1",
            status="ready",
            status_message="",
            total_chunks=42,
        )
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()


def test_update_doc_status_only_status_when_optionals_missing():
    repo, stub = _make_repo()
    stub.add_response(
        "update_item",
        {},
        expected_params={
            "TableName": TABLE,
            "Key": {"kbDocId": "kd_1", "sk": "META"},
            "UpdateExpression": "SET #s = :s, updatedAt = :u",
            "ExpressionAttributeNames": {"#s": "status"},
            "ExpressionAttributeValues": {":s": "chunking", ":u": _any_int()},
        },
    )
    try:
        repo.update_doc_status(kb_doc_id="kd_1", status="chunking")
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()


def test_write_chunks_uses_batch_writer_with_chunk_sks():
    repo, stub = _make_repo()

    # First, get_doc is called internally to denormalize docTitle/sourceType.
    stub.add_response(
        "get_item",
        {
            "Item": {
                "kbDocId": {"S": "kd_x"},
                "sk": {"S": "META"},
                "docTitle": {"S": "Doc Title"},
                "sourceType": {"S": "training"},
                "filename": {"S": "f.pdf"},
                "contentType": {"S": "application/pdf"},
                "sizeBytes": {"N": "1"},
                "s3Key": {"S": "kb/kd_x/f.pdf"},
                "status": {"S": "embedding"},
                "statusMessage": {"S": ""},
                "totalChunks": {"N": "0"},
                "uploadedBy": {"S": "u_a"},
                "createdAt": {"N": "1"},
                "updatedAt": {"N": "2"},
            }
        },
        expected_params={
            "TableName": TABLE,
            "Key": {"kbDocId": "kd_x", "sk": "META"},
        },
    )

    captured: list[dict] = []
    # Replace batch_writer with a fake context manager that records put_items.
    class FakeBatch:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *exc):
            return False

        def put_item(self_inner, Item):  # noqa: N803 — match boto signature
            captured.append(Item)

    with patch.object(repo._table, "batch_writer", return_value=FakeBatch()):
        repo.write_chunks(
            "kd_x",
            [
                ChunkRecord(
                    chunkIdx=0,
                    chunkText="hello",
                    chunkTokens=5,
                    embedding=[0.1, 0.2, 0.3],
                    pageNumber=1,
                    sectionTitle="Intro",
                ),
                ChunkRecord(
                    chunkIdx=1,
                    chunkText="world",
                    chunkTokens=3,
                    embedding=[0.4, 0.5, 0.6],
                ),
            ],
        )
    stub.deactivate()

    assert len(captured) == 2
    assert captured[0]["sk"] == "chunk#0000"
    assert captured[0]["chunkIdx"] == 0
    assert captured[0]["chunkText"] == "hello"
    assert captured[0]["docTitle"] == "Doc Title"
    assert captured[0]["sourceType"] == "training"
    assert captured[0]["pageNumber"] == 1
    assert captured[0]["sectionTitle"] == "Intro"
    assert captured[1]["sk"] == "chunk#0001"
    # Optional fields absent when None.
    assert "pageNumber" not in captured[1]
    assert "sectionTitle" not in captured[1]


def test_scan_all_chunks_paginates_and_hydrates_chunks():
    repo, stub = _make_repo()
    # Two pages: first returns one chunk + LastEvaluatedKey, second returns one.
    stub.add_response(
        "scan",
        {
            "Items": [
                {
                    "kbDocId": {"S": "kd_1"},
                    "sk": {"S": "chunk#0000"},
                    "chunkIdx": {"N": "0"},
                    "chunkText": {"S": "alpha"},
                    "chunkTokens": {"N": "10"},
                    "embedding": {"L": [{"N": "0.1"}, {"N": "0.2"}]},
                    "docTitle": {"S": "Doc1"},
                    "sourceType": {"S": "training"},
                    "pageNumber": {"N": "3"},
                    "sectionTitle": {"S": "Sec A"},
                    "createdAt": {"N": "1"},
                }
            ],
            "LastEvaluatedKey": {
                "kbDocId": {"S": "kd_1"},
                "sk": {"S": "chunk#0000"},
            },
        },
    )
    stub.add_response(
        "scan",
        {
            "Items": [
                {
                    "kbDocId": {"S": "kd_2"},
                    "sk": {"S": "chunk#0000"},
                    "chunkIdx": {"N": "0"},
                    "chunkText": {"S": "beta"},
                    "chunkTokens": {"N": "20"},
                    "embedding": {"L": [{"N": "0.3"}, {"N": "0.4"}]},
                    "docTitle": {"S": "Doc2"},
                    "sourceType": {"S": "protocol"},
                    "createdAt": {"N": "2"},
                }
            ]
        },
    )
    try:
        chunks = repo.scan_all_chunks()
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()

    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].kbDocId == "kd_1"
    assert chunks[0].chunkText == "alpha"
    assert chunks[0].chunkTokens == 10
    assert chunks[0].pageNumber == 3
    assert chunks[0].sectionTitle == "Sec A"
    assert chunks[0].embedding == [0.1, 0.2]
    assert chunks[1].kbDocId == "kd_2"
    assert chunks[1].pageNumber is None


def test_delete_doc_queries_chunks_and_batch_deletes_all():
    repo, stub = _make_repo()
    # Query returns two chunk sks.
    stub.add_response(
        "query",
        {
            "Items": [
                {"sk": {"S": "chunk#0000"}},
                {"sk": {"S": "chunk#0001"}},
            ]
        },
    )

    captured: list[dict] = []

    class FakeBatch:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *exc):
            return False

        def delete_item(self_inner, Key):  # noqa: N803
            captured.append(Key)

    try:
        with patch.object(repo._table, "batch_writer", return_value=FakeBatch()):
            repo.delete_doc("kd_del")
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()

    # META + 2 chunks = 3 deletes
    assert len(captured) == 3
    assert {"kbDocId": "kd_del", "sk": "META"} in captured
    assert {"kbDocId": "kd_del", "sk": "chunk#0000"} in captured
    assert {"kbDocId": "kd_del", "sk": "chunk#0001"} in captured


# ---------- helpers ----------


class _AnyInt:
    def __eq__(self, other):
        return isinstance(other, int | Decimal)

    def __repr__(self) -> str:  # pragma: no cover
        return "_AnyInt()"


def _any_int() -> _AnyInt:
    return _AnyInt()

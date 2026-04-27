from decimal import Decimal
from unittest.mock import patch

from botocore.stub import Stubber

from anna_chat.jobs_repo import JobsRepo, TranslateJob

TABLE = "anna-chat-test-jobs"


def _make_repo() -> tuple[JobsRepo, Stubber]:
    repo = JobsRepo(jobs_table=TABLE, region="us-east-1")
    client = repo._table.meta.client  # noqa: SLF001 — test-only access
    stub = Stubber(client)
    stub.activate()
    return repo, stub


def test_new_job_id_shape():
    job_id = JobsRepo._new_job_id()
    assert job_id.startswith("tj_")
    hex_part = job_id[3:]
    assert len(hex_part) == 16
    int(hex_part, 16)


def test_create_job_writes_pending_row_with_ttl():
    repo, stub = _make_repo()
    stub.add_response(
        "put_item",
        {},
        expected_params={
            "TableName": TABLE,
            "Item": {
                "userId": "u1",
                "jobId": "tj_fixed0123456789",
                "jobType": "translate",
                "status": "pending",
                "statusMessage": "",
                "sourceAttachmentId": "att_abc",
                "sourceFilename": "Crank et al 2021.pdf",
                "sourceContentType": "application/pdf",
                "targetLanguage": "es",
                "targetLanguageLabel": "Spanish",
                "inputTokensApprox": 0,
                "outputTokensApprox": 0,
                "createdAt": _any_int(),
                "updatedAt": _any_int(),
                "ttl": _any_int(),
            },
        },
    )
    try:
        job = repo.create_job(
            user_id="u1",
            source_attachment_id="att_abc",
            source_filename="Crank et al 2021.pdf",
            source_content_type="application/pdf",
            target_language="es",
            target_language_label="Spanish",
            job_id="tj_fixed0123456789",
        )
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()
    assert isinstance(job, TranslateJob)
    assert job.status == "pending"
    assert job.jobType == "translate"
    # ttl is roughly 7 days from now in epoch seconds — sanity-check the
    # order of magnitude rather than the exact value.
    assert job.ttl > 0
    assert job.outputDocxKey == ""


def test_get_job_casts_decimals_to_ints():
    repo, stub = _make_repo()
    stub.add_response(
        "get_item",
        {
            "Item": {
                "userId": {"S": "u1"},
                "jobId": {"S": "tj_abc"},
                "jobType": {"S": "translate"},
                "status": {"S": "ready"},
                "statusMessage": {"S": ""},
                "sourceAttachmentId": {"S": "att_a"},
                "sourceFilename": {"S": "f.pdf"},
                "sourceContentType": {"S": "application/pdf"},
                "targetLanguage": {"S": "es"},
                "targetLanguageLabel": {"S": "Spanish"},
                "outputDocxKey": {"S": "translations/tj_abc/f_es.docx"},
                "outputPdfKey": {"S": "translations/tj_abc/f_es.pdf"},
                "inputTokensApprox": {"N": "1234"},
                "outputTokensApprox": {"N": "987"},
                "createdAt": {"N": "1700000000000"},
                "updatedAt": {"N": "1700000010000"},
                "ttl": {"N": "1700604800"},
            }
        },
        expected_params={
            "TableName": TABLE,
            "Key": {"userId": "u1", "jobId": "tj_abc"},
        },
    )
    try:
        job = repo.get_job(user_id="u1", job_id="tj_abc")
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()
    assert job is not None
    assert job.status == "ready"
    assert isinstance(job.inputTokensApprox, int)
    assert job.inputTokensApprox == 1234
    assert job.outputTokensApprox == 987
    assert job.createdAt == 1700000000000
    assert job.outputDocxKey.endswith(".docx")
    assert job.outputPdfKey.endswith(".pdf")


def test_get_job_returns_none_when_missing():
    repo, stub = _make_repo()
    stub.add_response(
        "get_item",
        {},
        expected_params={
            "TableName": TABLE,
            "Key": {"userId": "u1", "jobId": "tj_missing"},
        },
    )
    try:
        job = repo.get_job(user_id="u1", job_id="tj_missing")
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()
    assert job is None


def test_list_jobs_queries_partition_key_descending():
    repo, stub = _make_repo()
    captured: list[dict] = []
    with patch.object(
        repo._table,
        "query",
        wraps=lambda **kwargs: (captured.append(kwargs) or {"Items": []}),
    ):
        items = repo.list_jobs(user_id="u1", limit=25)
    stub.deactivate()
    assert items == []
    assert len(captured) == 1
    assert captured[0]["Limit"] == 25
    assert captured[0]["ScanIndexForward"] is False
    key_expr = captured[0]["KeyConditionExpression"]
    assert key_expr.get_expression()["values"][0].name == "userId"
    assert key_expr.get_expression()["values"][1] == "u1"


def test_update_status_only_status_when_optionals_missing():
    repo, stub = _make_repo()
    stub.add_response(
        "update_item",
        {},
        expected_params={
            "TableName": TABLE,
            "Key": {"userId": "u1", "jobId": "tj_1"},
            "UpdateExpression": "SET #s = :s, updatedAt = :u",
            "ExpressionAttributeNames": {"#s": "status"},
            "ExpressionAttributeValues": {
                ":s": "translating",
                ":u": _any_int(),
            },
        },
    )
    try:
        repo.update_status(
            user_id="u1", job_id="tj_1", status="translating"
        )
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()


def test_update_status_includes_message_and_token_counts():
    repo, stub = _make_repo()
    stub.add_response(
        "update_item",
        {},
        expected_params={
            "TableName": TABLE,
            "Key": {"userId": "u1", "jobId": "tj_1"},
            "UpdateExpression": (
                "SET #s = :s, updatedAt = :u, statusMessage = :m, "
                "inputTokensApprox = :it, outputTokensApprox = :ot"
            ),
            "ExpressionAttributeNames": {"#s": "status"},
            "ExpressionAttributeValues": {
                ":s": "error",
                ":u": _any_int(),
                ":m": "BedrockThrottled",
                ":it": 5000,
                ":ot": 4000,
            },
        },
    )
    try:
        repo.update_status(
            user_id="u1",
            job_id="tj_1",
            status="error",
            status_message="BedrockThrottled",
            input_tokens_approx=5000,
            output_tokens_approx=4000,
        )
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()


def test_set_outputs_marks_ready_with_keys():
    repo, stub = _make_repo()
    stub.add_response(
        "update_item",
        {},
        expected_params={
            "TableName": TABLE,
            "Key": {"userId": "u1", "jobId": "tj_1"},
            "UpdateExpression": (
                "SET #s = :s, updatedAt = :u, statusMessage = :m, "
                "outputDocxKey = :dk, outputPdfKey = :pk, "
                "outputTokensApprox = :ot"
            ),
            "ExpressionAttributeNames": {"#s": "status"},
            "ExpressionAttributeValues": {
                ":s": "ready",
                ":u": _any_int(),
                ":m": "",
                ":dk": "translations/tj_1/f_es.docx",
                ":pk": "translations/tj_1/f_es.pdf",
                ":ot": 800,
            },
        },
    )
    try:
        repo.set_outputs(
            user_id="u1",
            job_id="tj_1",
            output_docx_key="translations/tj_1/f_es.docx",
            output_pdf_key="translations/tj_1/f_es.pdf",
            output_tokens_approx=800,
        )
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()


# ---------- helpers ----------


class _AnyInt:
    def __eq__(self, other):
        return isinstance(other, int | Decimal)

    def __repr__(self) -> str:  # pragma: no cover
        return "_AnyInt()"


def _any_int() -> _AnyInt:
    return _AnyInt()

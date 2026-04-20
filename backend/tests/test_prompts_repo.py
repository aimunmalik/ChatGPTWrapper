from decimal import Decimal

from botocore.exceptions import ClientError
from botocore.stub import Stubber

from anna_chat.prompts_repo import Prompt, PromptsRepo

TABLE = "anna-chat-test-prompts"


def _make_repo() -> tuple[PromptsRepo, Stubber]:
    repo = PromptsRepo(prompts_table=TABLE, region="us-east-1")
    client = repo._table.meta.client  # noqa: SLF001 — test-only access
    stub = Stubber(client)
    stub.activate()
    return repo, stub


def test_new_prompt_id_has_expected_shape():
    pid = PromptsRepo._new_prompt_id()
    assert pid.startswith("p_")
    # 16 hex chars after the "p_" prefix
    hex_part = pid[2:]
    assert len(hex_part) == 16
    int(hex_part, 16)  # raises if not hex


def test_create_puts_expected_item_and_returns_dataclass():
    repo, stub = _make_repo()
    stub.add_response(
        "put_item",
        {},
        expected_params={
            "TableName": TABLE,
            "Item": {
                "userId": "u1",
                "promptId": "p_fixed0123456789",
                "title": "My template",
                "body": "Do the thing",
                "createdAt": _any_int(),
                "updatedAt": _any_int(),
            },
        },
    )
    try:
        prompt = repo.create(
            user_id="u1",
            title="My template",
            body="Do the thing",
            prompt_id="p_fixed0123456789",
        )
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()

    assert isinstance(prompt, Prompt)
    assert prompt.promptId == "p_fixed0123456789"
    assert prompt.userId == "u1"
    assert prompt.title == "My template"
    assert prompt.body == "Do the thing"
    assert prompt.createdAt == prompt.updatedAt
    assert isinstance(prompt.createdAt, int)


def test_list_for_user_returns_prompts_with_ints():
    repo, stub = _make_repo()
    stub.add_response(
        "query",
        {
            "Items": [
                {
                    "userId": {"S": "u1"},
                    "promptId": {"S": "p_aaaaaaaaaaaaaaaa"},
                    "title": {"S": "T1"},
                    "body": {"S": "B1"},
                    "createdAt": {"N": "1700000000000"},
                    "updatedAt": {"N": "1700000001000"},
                },
                {
                    "userId": {"S": "u1"},
                    "promptId": {"S": "p_bbbbbbbbbbbbbbbb"},
                    "title": {"S": "T2"},
                    "body": {"S": "B2"},
                    "createdAt": {"N": "1700000002000"},
                    "updatedAt": {"N": "1700000002000"},
                },
            ]
        },
    )
    try:
        prompts = repo.list_for_user(user_id="u1")
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()

    assert len(prompts) == 2
    assert prompts[0].promptId == "p_aaaaaaaaaaaaaaaa"
    assert prompts[0].title == "T1"
    assert isinstance(prompts[0].createdAt, int)
    assert prompts[0].createdAt == 1700000000000
    assert prompts[1].title == "T2"
    assert prompts[1].updatedAt == 1700000002000


def test_get_returns_prompt_when_present():
    repo, stub = _make_repo()
    stub.add_response(
        "get_item",
        {
            "Item": {
                "userId": {"S": "u1"},
                "promptId": {"S": "p_abc"},
                "title": {"S": "T"},
                "body": {"S": "B"},
                "createdAt": {"N": "1700000000000"},
                "updatedAt": {"N": "1700000000500"},
            }
        },
        expected_params={
            "TableName": TABLE,
            "Key": {"userId": "u1", "promptId": "p_abc"},
        },
    )
    try:
        prompt = repo.get(user_id="u1", prompt_id="p_abc")
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()

    assert prompt is not None
    assert prompt.promptId == "p_abc"
    assert prompt.title == "T"
    assert prompt.body == "B"
    assert prompt.createdAt == 1700000000000
    assert prompt.updatedAt == 1700000000500


def test_get_returns_none_when_missing():
    repo, stub = _make_repo()
    stub.add_response(
        "get_item",
        {},
        expected_params={
            "TableName": TABLE,
            "Key": {"userId": "u1", "promptId": "p_missing"},
        },
    )
    try:
        prompt = repo.get(user_id="u1", prompt_id="p_missing")
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()
    assert prompt is None


def test_update_returns_prompt_with_new_fields():
    repo, stub = _make_repo()
    stub.add_response(
        "update_item",
        {
            "Attributes": {
                "userId": {"S": "u1"},
                "promptId": {"S": "p_xyz"},
                "title": {"S": "New title"},
                "body": {"S": "New body"},
                "createdAt": {"N": "1700000000000"},
                "updatedAt": {"N": "1800000000000"},
            }
        },
    )
    try:
        prompt = repo.update(
            user_id="u1",
            prompt_id="p_xyz",
            title="New title",
            body="New body",
        )
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()

    assert prompt is not None
    assert prompt.promptId == "p_xyz"
    assert prompt.title == "New title"
    assert prompt.body == "New body"
    assert prompt.updatedAt == 1800000000000


def test_update_returns_none_on_conditional_check_failure():
    repo, stub = _make_repo()
    stub.add_client_error(
        "update_item",
        service_error_code="ConditionalCheckFailedException",
        service_message="The conditional request failed",
        http_status_code=400,
    )
    try:
        result = repo.update(
            user_id="u1",
            prompt_id="p_nope",
            title="x",
            body="y",
        )
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()
    assert result is None


def test_delete_returns_true_when_row_existed():
    repo, stub = _make_repo()
    stub.add_response(
        "delete_item",
        {
            "Attributes": {
                "userId": {"S": "u1"},
                "promptId": {"S": "p_del"},
                "title": {"S": "T"},
                "body": {"S": "B"},
                "createdAt": {"N": "1"},
                "updatedAt": {"N": "1"},
            }
        },
        expected_params={
            "TableName": TABLE,
            "Key": {"userId": "u1", "promptId": "p_del"},
            "ReturnValues": "ALL_OLD",
        },
    )
    try:
        existed = repo.delete(user_id="u1", prompt_id="p_del")
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()
    assert existed is True


def test_delete_returns_false_when_no_row():
    repo, stub = _make_repo()
    stub.add_response(
        "delete_item",
        {},
        expected_params={
            "TableName": TABLE,
            "Key": {"userId": "u1", "promptId": "p_missing"},
            "ReturnValues": "ALL_OLD",
        },
    )
    try:
        existed = repo.delete(user_id="u1", prompt_id="p_missing")
        stub.assert_no_pending_responses()
    finally:
        stub.deactivate()
    assert existed is False


# ---------- helpers ----------


class _AnyInt:
    def __eq__(self, other):
        return isinstance(other, int | Decimal)

    def __repr__(self) -> str:  # pragma: no cover
        return "_AnyInt()"


def _any_int() -> _AnyInt:
    return _AnyInt()


# Keep ClientError import resolvable for tooling that checks unused imports.
_ = ClientError

"""Tests for handlers/admin_users.py.

Uses ``botocore.stub.Stubber`` against the real ``cognito-idp`` client the
handler module exposes via ``_cognito()``. This mirrors how ``test_kb_repo.py``
stubs the DynamoDB table client.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import boto3
import pytest
from botocore.stub import Stubber

from anna_chat.handlers import admin_users
from anna_chat.settings import Settings

POOL_ID = "us-east-1_TESTPOOL"
ADMIN_SUB = "u_admin_caller"
OTHER_SUB = "u_other_user"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _settings() -> Settings:
    return Settings(
        aws_region="us-east-1",
        cognito_user_pool_id=POOL_ID,
        cognito_spa_client_id="client-id",
        conversations_table="t-conv",
        messages_table="t-msg",
        bedrock_model_id="any",
        message_ttl_days=90,
        attachments_table="t-att",
        attachments_bucket="b-att",
        attachments_max_size_bytes=52428800,
        attachments_max_text_bytes=512000,
        prompts_table="t-prompts",
        kb_table="t-kb",
        kb_bucket="b-kb",
        kb_max_size_bytes=104857600,
        jobs_table="t-jobs",
        translate_worker_function_name="lambda_translate_worker",
    )


@pytest.fixture(autouse=True)
def _patch_settings_and_cognito(monkeypatch):
    """Replace the cached Cognito client + Settings with stub-friendly ones.

    The ``lru_cache``-wrapped ``_settings()`` and ``_cognito()`` would
    otherwise leak between tests. We clear the caches and inject a fresh
    real client per test so every test gets its own ``Stubber``.
    """
    admin_users._settings.cache_clear()
    admin_users._cognito.cache_clear()

    settings = _settings()
    monkeypatch.setattr(admin_users, "_settings", lambda: settings)

    client = boto3.client("cognito-idp", region_name="us-east-1")
    monkeypatch.setattr(admin_users, "_cognito", lambda: client)

    yield client

    # Monkeypatch restores the original lru_cache-wrapped functions after
    # yield returns, so we don't need to clear caches again here.


def _admin_event(
    *,
    route_key: str,
    sub: str = ADMIN_SUB,
    path_params: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    groups: str = '["admins"]',
) -> dict[str, Any]:
    return {
        "routeKey": route_key,
        "pathParameters": path_params or {},
        "body": json.dumps(body) if body is not None else None,
        "isBase64Encoded": False,
        "requestContext": {
            "authorizer": {
                "jwt": {
                    "claims": {
                        "sub": sub,
                        "email": "admin@example.com",
                        "name": "Admin",
                        "cognito:groups": groups,
                    }
                }
            }
        },
    }


def _cognito_user(
    *,
    username: str,
    email: str,
    name: str = "Some User",
    enabled: bool = True,
    status: str = "CONFIRMED",
) -> dict[str, Any]:
    return {
        "Username": username,
        "Attributes": [
            {"Name": "sub", "Value": username},
            {"Name": "email", "Value": email},
            {"Name": "name", "Value": name},
        ],
        "UserCreateDate": dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        "UserLastModifiedDate": dt.datetime(2026, 2, 1, tzinfo=dt.UTC),
        "Enabled": enabled,
        "UserStatus": status,
    }


def _admin_get_user_response(
    *,
    username: str,
    email: str,
    name: str = "Some User",
    enabled: bool = True,
    status: str = "CONFIRMED",
) -> dict[str, Any]:
    return {
        "Username": username,
        "UserAttributes": [
            {"Name": "sub", "Value": username},
            {"Name": "email", "Value": email},
            {"Name": "name", "Value": name},
        ],
        "UserCreateDate": dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        "UserLastModifiedDate": dt.datetime(2026, 2, 1, tzinfo=dt.UTC),
        "Enabled": enabled,
        "UserStatus": status,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_list_users_merges_admin_group_membership(_patch_settings_and_cognito):
    client = _patch_settings_and_cognito
    stub = Stubber(client)

    # First the admin-group lookup.
    stub.add_response(
        "list_users_in_group",
        {"Users": [{"Username": ADMIN_SUB}]},
        expected_params={
            "UserPoolId": POOL_ID,
            "GroupName": "admins",
            "Limit": 60,
        },
    )
    # Then the user list.
    stub.add_response(
        "list_users",
        {
            "Users": [
                _cognito_user(
                    username=ADMIN_SUB,
                    email="admin@example.com",
                    name="Admin",
                ),
                _cognito_user(
                    username="microsoft_abc123",
                    email="clinician@example.com",
                    name="Clinician",
                    status="EXTERNAL_PROVIDER",
                ),
            ]
        },
        expected_params={"UserPoolId": POOL_ID, "Limit": 60},
    )

    with stub:
        resp = admin_users.handler(
            _admin_event(route_key="GET /admin/users"), None
        )

    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert len(body["users"]) == 2
    by_user = {u["username"]: u for u in body["users"]}
    assert by_user[ADMIN_SUB]["isAdmin"] is True
    assert by_user[ADMIN_SUB]["identitySource"] == "local"
    assert by_user[ADMIN_SUB]["email"] == "admin@example.com"
    assert by_user["microsoft_abc123"]["isAdmin"] is False
    assert by_user["microsoft_abc123"]["identitySource"] == "microsoft"
    assert by_user["microsoft_abc123"]["status"] == "EXTERNAL_PROVIDER"


def test_invite_user_creates_and_promotes_when_isAdmin_true(
    _patch_settings_and_cognito,
):
    client = _patch_settings_and_cognito
    stub = Stubber(client)

    stub.add_response(
        "admin_create_user",
        {
            "User": _cognito_user(
                username="new@example.com",
                email="new@example.com",
                name="New Person",
                status="FORCE_CHANGE_PASSWORD",
            )
        },
        expected_params={
            "UserPoolId": POOL_ID,
            "Username": "new@example.com",
            "UserAttributes": [
                {"Name": "email", "Value": "new@example.com"},
                {"Name": "email_verified", "Value": "true"},
                {"Name": "name", "Value": "New Person"},
            ],
            "DesiredDeliveryMediums": ["EMAIL"],
        },
    )
    stub.add_response(
        "admin_add_user_to_group",
        {},
        expected_params={
            "UserPoolId": POOL_ID,
            "Username": "new@example.com",
            "GroupName": "admins",
        },
    )

    with stub:
        resp = admin_users.handler(
            _admin_event(
                route_key="POST /admin/users",
                body={
                    "email": "new@example.com",
                    "name": "New Person",
                    "isAdmin": True,
                },
            ),
            None,
        )

    assert resp["statusCode"] == 201
    body = json.loads(resp["body"])
    assert body["username"] == "new@example.com"
    assert body["email"] == "new@example.com"
    assert body["name"] == "New Person"
    assert body["isAdmin"] is True


def test_invite_user_skips_group_add_when_not_admin(
    _patch_settings_and_cognito,
):
    client = _patch_settings_and_cognito
    stub = Stubber(client)

    stub.add_response(
        "admin_create_user",
        {
            "User": _cognito_user(
                username="plain@example.com",
                email="plain@example.com",
                name="Plain",
                status="FORCE_CHANGE_PASSWORD",
            )
        },
    )
    # No add-to-group call queued — Stubber would explode if one fired.

    with stub:
        resp = admin_users.handler(
            _admin_event(
                route_key="POST /admin/users",
                body={
                    "email": "plain@example.com",
                    "name": "Plain",
                    "isAdmin": False,
                },
            ),
            None,
        )

    assert resp["statusCode"] == 201
    body = json.loads(resp["body"])
    assert body["isAdmin"] is False


def test_invite_user_duplicate_email_returns_409(_patch_settings_and_cognito):
    client = _patch_settings_and_cognito
    stub = Stubber(client)

    stub.add_client_error(
        "admin_create_user",
        service_error_code="UsernameExistsException",
        service_message="User account already exists",
        http_status_code=400,
    )

    with stub:
        resp = admin_users.handler(
            _admin_event(
                route_key="POST /admin/users",
                body={
                    "email": "dup@example.com",
                    "name": "Dup",
                    "isAdmin": False,
                },
            ),
            None,
        )

    assert resp["statusCode"] == 409
    assert "already exists" in json.loads(resp["body"])["error"]


def test_invite_user_validates_required_fields(_patch_settings_and_cognito):
    resp = admin_users.handler(
        _admin_event(
            route_key="POST /admin/users",
            body={"email": "", "name": "X", "isAdmin": False},
        ),
        None,
    )
    assert resp["statusCode"] == 400
    assert "email" in json.loads(resp["body"])["error"]


def test_update_user_disable_also_signs_out(_patch_settings_and_cognito):
    client = _patch_settings_and_cognito
    stub = Stubber(client)

    # admin_get_user (current state: enabled=True)
    stub.add_response(
        "admin_get_user",
        _admin_get_user_response(
            username=OTHER_SUB, email="other@example.com", enabled=True
        ),
        expected_params={"UserPoolId": POOL_ID, "Username": OTHER_SUB},
    )
    stub.add_response(
        "admin_disable_user",
        {},
        expected_params={"UserPoolId": POOL_ID, "Username": OTHER_SUB},
    )
    stub.add_response(
        "admin_user_global_sign_out",
        {},
        expected_params={"UserPoolId": POOL_ID, "Username": OTHER_SUB},
    )
    # No group change → still need a list_users_in_group for the response.
    stub.add_response(
        "list_users_in_group",
        {"Users": [{"Username": ADMIN_SUB}]},
        expected_params={
            "UserPoolId": POOL_ID,
            "GroupName": "admins",
            "Limit": 60,
        },
    )
    # Refresh state at the end.
    stub.add_response(
        "admin_get_user",
        _admin_get_user_response(
            username=OTHER_SUB, email="other@example.com", enabled=False
        ),
        expected_params={"UserPoolId": POOL_ID, "Username": OTHER_SUB},
    )

    with stub:
        resp = admin_users.handler(
            _admin_event(
                route_key="PATCH /admin/users/{username}",
                path_params={"username": OTHER_SUB},
                body={"enabled": False},
            ),
            None,
        )

    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert body["enabled"] is False
    assert body["isAdmin"] is False


def test_update_user_promote_to_admin_round_trip(_patch_settings_and_cognito):
    client = _patch_settings_and_cognito
    stub = Stubber(client)

    stub.add_response(
        "admin_get_user",
        _admin_get_user_response(
            username=OTHER_SUB, email="other@example.com", enabled=True
        ),
    )
    # is_admin toggle path: list group to check current membership.
    stub.add_response(
        "list_users_in_group",
        {"Users": [{"Username": ADMIN_SUB}]},
    )
    stub.add_response(
        "admin_add_user_to_group",
        {},
        expected_params={
            "UserPoolId": POOL_ID,
            "Username": OTHER_SUB,
            "GroupName": "admins",
        },
    )
    stub.add_response(
        "admin_get_user",
        _admin_get_user_response(
            username=OTHER_SUB, email="other@example.com", enabled=True
        ),
    )

    with stub:
        resp = admin_users.handler(
            _admin_event(
                route_key="PATCH /admin/users/{username}",
                path_params={"username": OTHER_SUB},
                body={"isAdmin": True},
            ),
            None,
        )

    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert body["isAdmin"] is True


def test_update_user_demote_admin_when_not_last(_patch_settings_and_cognito):
    client = _patch_settings_and_cognito
    stub = Stubber(client)

    stub.add_response(
        "admin_get_user",
        _admin_get_user_response(
            username=OTHER_SUB, email="other@example.com"
        ),
    )
    # Two admins in the group → safe to remove one.
    stub.add_response(
        "list_users_in_group",
        {
            "Users": [
                {"Username": ADMIN_SUB},
                {"Username": OTHER_SUB},
            ]
        },
    )
    stub.add_response(
        "admin_remove_user_from_group",
        {},
        expected_params={
            "UserPoolId": POOL_ID,
            "Username": OTHER_SUB,
            "GroupName": "admins",
        },
    )
    stub.add_response(
        "admin_get_user",
        _admin_get_user_response(
            username=OTHER_SUB, email="other@example.com"
        ),
    )

    with stub:
        resp = admin_users.handler(
            _admin_event(
                route_key="PATCH /admin/users/{username}",
                path_params={"username": OTHER_SUB},
                body={"isAdmin": False},
            ),
            None,
        )

    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert body["isAdmin"] is False


def test_self_disable_returns_403_with_errorType(_patch_settings_and_cognito):
    resp = admin_users.handler(
        _admin_event(
            route_key="PATCH /admin/users/{username}",
            path_params={"username": ADMIN_SUB},
            body={"enabled": False},
        ),
        None,
    )
    assert resp["statusCode"] == 403
    body = json.loads(resp["body"])
    assert body["errorType"] == "SelfDisable"
    assert "disable" in body["error"].lower()


def test_self_demote_returns_403_with_errorType(_patch_settings_and_cognito):
    resp = admin_users.handler(
        _admin_event(
            route_key="PATCH /admin/users/{username}",
            path_params={"username": ADMIN_SUB},
            body={"isAdmin": False},
        ),
        None,
    )
    assert resp["statusCode"] == 403
    body = json.loads(resp["body"])
    assert body["errorType"] == "SelfDemote"


def test_last_admin_removal_blocked_with_errorType(
    _patch_settings_and_cognito,
):
    client = _patch_settings_and_cognito
    stub = Stubber(client)

    # The target is some OTHER admin (not the caller) so SelfDemote doesn't
    # short-circuit. They are the only member of the group.
    stub.add_response(
        "admin_get_user",
        _admin_get_user_response(
            username=OTHER_SUB, email="other@example.com"
        ),
    )
    stub.add_response(
        "list_users_in_group",
        {"Users": [{"Username": OTHER_SUB}]},
    )

    with stub:
        resp = admin_users.handler(
            _admin_event(
                route_key="PATCH /admin/users/{username}",
                path_params={"username": OTHER_SUB},
                body={"isAdmin": False},
            ),
            None,
        )

    assert resp["statusCode"] == 403
    body = json.loads(resp["body"])
    assert body["errorType"] == "LastAdmin"


def test_update_user_404_when_target_missing(_patch_settings_and_cognito):
    client = _patch_settings_and_cognito
    stub = Stubber(client)

    stub.add_client_error(
        "admin_get_user",
        service_error_code="UserNotFoundException",
        service_message="User does not exist.",
        http_status_code=400,
    )

    with stub:
        resp = admin_users.handler(
            _admin_event(
                route_key="PATCH /admin/users/{username}",
                path_params={"username": "ghost"},
                body={"enabled": True},
            ),
            None,
        )

    assert resp["statusCode"] == 404


def test_sign_out_happy_path(_patch_settings_and_cognito):
    client = _patch_settings_and_cognito
    stub = Stubber(client)

    stub.add_response(
        "admin_user_global_sign_out",
        {},
        expected_params={"UserPoolId": POOL_ID, "Username": OTHER_SUB},
    )

    with stub:
        resp = admin_users.handler(
            _admin_event(
                route_key="POST /admin/users/{username}/sign-out",
                path_params={"username": OTHER_SUB},
            ),
            None,
        )

    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert body == {"username": OTHER_SUB, "signedOut": True}


def test_sign_out_404_when_target_missing(_patch_settings_and_cognito):
    client = _patch_settings_and_cognito
    stub = Stubber(client)

    stub.add_client_error(
        "admin_user_global_sign_out",
        service_error_code="UserNotFoundException",
        service_message="User does not exist.",
        http_status_code=400,
    )

    with stub:
        resp = admin_users.handler(
            _admin_event(
                route_key="POST /admin/users/{username}/sign-out",
                path_params={"username": "ghost"},
            ),
            None,
        )

    assert resp["statusCode"] == 404


def test_non_admin_caller_blocked(_patch_settings_and_cognito):
    resp = admin_users.handler(
        _admin_event(
            route_key="GET /admin/users",
            groups='["users"]',
        ),
        None,
    )
    assert resp["statusCode"] == 403
    assert "admin" in json.loads(resp["body"])["error"]


def test_unknown_route_returns_404(_patch_settings_and_cognito):
    resp = admin_users.handler(
        _admin_event(route_key="DELETE /admin/users/{username}"),
        None,
    )
    assert resp["statusCode"] == 404

import json

import pytest

from anna_chat.http import (
    AuthenticatedUser,
    HttpError,
    authenticate,
    error,
    ok,
    parse_json_body,
)
from anna_chat.settings import Settings


def _settings() -> Settings:
    return Settings(
        aws_region="us-east-1",
        cognito_user_pool_id="us-east-1_xxx",
        cognito_spa_client_id="client-id",
        conversations_table="t-conv",
        messages_table="t-msg",
        bedrock_model_id="any",
        message_ttl_days=90,
    )


def test_ok_returns_json_with_no_store():
    resp = ok({"hello": "world"})
    assert resp["statusCode"] == 200
    assert resp["headers"]["Content-Type"] == "application/json"
    assert resp["headers"]["Cache-Control"] == "no-store"
    assert json.loads(resp["body"]) == {"hello": "world"}


def test_error_wraps_message():
    resp = error(404, "not found")
    assert resp["statusCode"] == 404
    assert json.loads(resp["body"]) == {"error": "not found"}


def test_parse_json_body_valid():
    event = {"body": '{"message": "hi"}', "isBase64Encoded": False}
    assert parse_json_body(event) == {"message": "hi"}


def test_parse_json_body_missing():
    assert parse_json_body({}) == {}


def test_parse_json_body_invalid_json():
    with pytest.raises(HttpError) as exc:
        parse_json_body({"body": "{not json"})
    assert exc.value.status == 400


def test_parse_json_body_array_rejected():
    with pytest.raises(HttpError) as exc:
        parse_json_body({"body": "[1,2,3]"})
    assert exc.value.status == 400


def test_authenticate_reads_claims_from_authorizer():
    event = {
        "requestContext": {
            "authorizer": {
                "jwt": {
                    "claims": {
                        "sub": "u_123",
                        "email": "user@example.com",
                        "name": "Some User",
                        "cognito:groups": '["admins","users"]',
                    }
                }
            }
        }
    }
    user = authenticate(event, _settings())
    assert isinstance(user, AuthenticatedUser)
    assert user.sub == "u_123"
    assert user.email == "user@example.com"
    assert user.name == "Some User"
    assert set(user.groups) == {"admins", "users"}


def test_authenticate_handles_list_groups():
    event = {
        "requestContext": {
            "authorizer": {
                "jwt": {"claims": {"sub": "u_1", "cognito:groups": ["users"]}}
            }
        }
    }
    user = authenticate(event, _settings())
    assert user.groups == ("users",)


def test_authenticate_missing_claims_raises_401():
    with pytest.raises(HttpError) as exc:
        authenticate({}, _settings())
    assert exc.value.status == 401

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from anna_chat.logging_config import get_logger
from anna_chat.settings import Settings

logger = get_logger(__name__)


class _DynamoJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Decimal):
            if o % 1 == 0:
                return int(o)
            return float(o)
        return super().default(o)


class HttpError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class AuthenticatedUser:
    sub: str
    email: str
    name: str
    groups: tuple[str, ...]


def ok(body: dict[str, Any], status: int = 200) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "no-store",
        },
        "body": json.dumps(body, cls=_DynamoJSONEncoder),
    }


def error(status: int, message: str) -> dict[str, Any]:
    return ok({"error": message}, status=status)


def parse_json_body(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        import base64

        raw = base64.b64decode(raw).decode("utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HttpError(400, f"invalid json: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise HttpError(400, "body must be a JSON object")
    return parsed


def authenticate(event: dict[str, Any], settings: Settings) -> AuthenticatedUser:
    claims = (
        event.get("requestContext", {})
        .get("authorizer", {})
        .get("jwt", {})
        .get("claims", {})
    )

    sub = claims.get("sub")
    if not sub:
        raise HttpError(401, "missing authorizer claims")

    groups_raw = claims.get("cognito:groups", "")
    if isinstance(groups_raw, str):
        groups_raw = groups_raw.strip("[]")
        groups = tuple(g.strip().strip('"') for g in groups_raw.split(",") if g.strip())
    elif isinstance(groups_raw, list):
        groups = tuple(groups_raw)
    else:
        groups = ()

    _ = settings
    return AuthenticatedUser(
        sub=sub,
        email=claims.get("email", ""),
        name=claims.get("name", ""),
        groups=groups,
    )


def require_admin(user: AuthenticatedUser) -> None:
    """Enforce that `user` belongs to the Cognito `admins` group.

    Raises HttpError(403) when the user is authenticated but not an admin.
    """
    if "admins" not in user.groups:
        raise HttpError(403, "admin only")

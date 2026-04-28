"""HTTP handler for the admin-only user-management routes.

Routes:
  - GET   /admin/users
  - POST  /admin/users
  - PATCH /admin/users/{username}
  - POST  /admin/users/{username}/sign-out

Every route requires the authenticated caller to be in the Cognito `admins`
group. See docs/ADMIN_USERS_CONTRACT.md for the wire format and self-protection
rules (SelfDisable, SelfDemote, LastAdmin).

Cognito is reached through a single boto3 `cognito-idp` client exposed via
``_cognito()`` so tests can swap a ``botocore.stub.Stubber`` underneath, the
same way ``handlers/kb.py`` exposes ``_s3()``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import boto3
from botocore.exceptions import ClientError

from anna_chat.http import (
    AuthenticatedUser,
    HttpError,
    authenticate,
    error,
    ok,
    parse_json_body,
    require_admin,
)
from anna_chat.logging_config import configure_logging, get_logger
from anna_chat.settings import Settings

configure_logging()
logger = get_logger(__name__)

ADMIN_GROUP = "admins"
LIST_USERS_LIMIT = 60  # Cognito max per page
LIST_USERS_HARD_CAP = 200  # contract: defensive cap before MVP needs paging


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def _cognito():
    s = _settings()
    return boto3.client("cognito-idp", region_name=s.aws_region)


# ---------------------------------------------------------------------------
# Cognito attribute helpers
# ---------------------------------------------------------------------------


def _attrs_to_dict(attributes: list[dict[str, Any]] | None) -> dict[str, str]:
    """Flatten Cognito's ``[{Name, Value}, ...]`` shape into a plain dict."""
    if not attributes:
        return {}
    return {a.get("Name", ""): a.get("Value", "") for a in attributes}


def _identity_source(username: str) -> str:
    """Derive the identity source from the username prefix.

    Federated Microsoft users get a ``microsoft_<sub>`` username from Cognito;
    everyone else is a local pool user. Cheaper and more accurate than calling
    ``AdminListGroupsForUser`` per row.
    """
    return "microsoft" if username.startswith("microsoft_") else "local"


def _epoch_ms(dt: Any) -> int | None:
    """Cognito returns timezone-aware datetimes; flatten to epoch millis."""
    if dt is None:
        return None
    try:
        return int(dt.timestamp() * 1000)
    except Exception:  # pragma: no cover — paranoia for stub edge cases
        return None


def _user_response(
    cognito_user: dict[str, Any],
    *,
    is_admin: bool,
) -> dict[str, Any]:
    """Project a Cognito user record into the API contract shape."""
    username = cognito_user.get("Username", "")
    attrs = _attrs_to_dict(
        # ``list_users`` returns ``Attributes``; ``admin_get_user`` /
        # ``admin_create_user`` return ``UserAttributes``.
        cognito_user.get("Attributes") or cognito_user.get("UserAttributes")
    )
    return {
        "username": username,
        "email": attrs.get("email", ""),
        "name": attrs.get("name", ""),
        "status": cognito_user.get("UserStatus", ""),
        "enabled": bool(cognito_user.get("Enabled", False)),
        "isAdmin": is_admin,
        "identitySource": _identity_source(username),
        "createdAt": _epoch_ms(cognito_user.get("UserCreateDate")),
        "lastModifiedAt": _epoch_ms(cognito_user.get("UserLastModifiedDate")),
    }


# ---------------------------------------------------------------------------
# Handler entrypoint
# ---------------------------------------------------------------------------


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    try:
        user = authenticate(event, _settings())
        require_admin(user)

        route_key = event.get("routeKey", "")
        path_params = event.get("pathParameters") or {}

        if route_key == "GET /admin/users":
            return _list_users(user)

        if route_key == "POST /admin/users":
            return _invite_user(event, user)

        if route_key == "PATCH /admin/users/{username}":
            target = path_params.get("username", "")
            return _update_user(event, user, target)

        if route_key == "POST /admin/users/{username}/sign-out":
            target = path_params.get("username", "")
            return _sign_out_user(user, target)

        return error(404, "route not found")

    except HttpError as exc:
        logger.info(
            "admin_users_http_error",
            extra={
                "status": exc.status,
                "reason": exc.message,
                "errorType": exc.error_type,
            },
        )
        return error(exc.status, exc.message, error_type=exc.error_type)
    except ClientError as exc:
        # Surface AWS error code (PHI-safe) without leaking the message,
        # which can echo ARNs / resource names.
        logger.error(
            "admin_users_unhandled_error",
            extra={
                "errorType": type(exc).__name__,
                "awsErrorCode": exc.response.get("Error", {}).get("Code", ""),
            },
        )
        return error(500, "internal error")
    except Exception as exc:
        logger.error(
            "admin_users_unhandled_error",
            extra={"errorType": type(exc).__name__},
        )
        return error(500, "internal error")


# ---------------------------------------------------------------------------
# Cognito wrappers
# ---------------------------------------------------------------------------


def _admin_get_user(username: str) -> dict[str, Any]:
    """Fetch a single user, mapping ``UserNotFoundException`` to 404."""
    settings = _settings()
    try:
        return _cognito().admin_get_user(
            UserPoolId=settings.cognito_user_pool_id,
            Username=username,
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code == "UserNotFoundException":
            raise HttpError(404, "user not found") from exc
        raise


def _list_admin_usernames() -> set[str]:
    """All usernames currently in the ``admins`` group, paginated.

    One round trip (per page) instead of N ``AdminListGroupsForUser`` calls
    while computing ``isAdmin`` for the list response.
    """
    settings = _settings()
    cognito = _cognito()
    usernames: set[str] = set()
    next_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {
            "UserPoolId": settings.cognito_user_pool_id,
            "GroupName": ADMIN_GROUP,
            "Limit": LIST_USERS_LIMIT,
        }
        if next_token:
            kwargs["NextToken"] = next_token
        resp = cognito.list_users_in_group(**kwargs)
        for u in resp.get("Users", []) or []:
            uname = u.get("Username")
            if uname:
                usernames.add(uname)
        next_token = resp.get("NextToken")
        if not next_token:
            break
    return usernames


# ---------------------------------------------------------------------------
# Route implementations
# ---------------------------------------------------------------------------


def _list_users(actor: AuthenticatedUser) -> dict[str, Any]:
    """List every user in the pool (capped) merged with admin-group state."""
    settings = _settings()
    cognito = _cognito()

    admin_usernames = _list_admin_usernames()

    users: list[dict[str, Any]] = []
    next_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {
            "UserPoolId": settings.cognito_user_pool_id,
            "Limit": LIST_USERS_LIMIT,
        }
        if next_token:
            kwargs["PaginationToken"] = next_token
        resp = cognito.list_users(**kwargs)
        for u in resp.get("Users", []) or []:
            users.append(
                _user_response(
                    u,
                    is_admin=u.get("Username", "") in admin_usernames,
                )
            )
            if len(users) >= LIST_USERS_HARD_CAP:
                break
        next_token = resp.get("PaginationToken")
        if not next_token or len(users) >= LIST_USERS_HARD_CAP:
            break

    logger.info(
        "admin_users_listed",
        extra={"actorUserId": actor.sub, "count": len(users)},
    )
    return ok({"users": users})


def _invite_user(
    event: dict[str, Any], actor: AuthenticatedUser
) -> dict[str, Any]:
    """Create a new Cognito user; optionally promote to admins.

    Cognito mails the temporary password directly; we never see or log it.
    """
    body = parse_json_body(event)
    email_raw = body.get("email")
    name_raw = body.get("name")
    is_admin_raw = body.get("isAdmin", False)

    if not isinstance(email_raw, str) or not email_raw.strip():
        raise HttpError(400, "email is required")
    if not isinstance(name_raw, str) or not name_raw.strip():
        raise HttpError(400, "name is required")
    if not isinstance(is_admin_raw, bool):
        raise HttpError(400, "isAdmin must be a boolean")

    email = email_raw.strip()
    name = name_raw.strip()
    if "@" not in email or " " in email:
        raise HttpError(400, "email is malformed")

    settings = _settings()
    cognito = _cognito()

    try:
        created = cognito.admin_create_user(
            UserPoolId=settings.cognito_user_pool_id,
            Username=email,
            UserAttributes=[
                {"Name": "email", "Value": email},
                {"Name": "email_verified", "Value": "true"},
                {"Name": "name", "Value": name},
            ],
            DesiredDeliveryMediums=["EMAIL"],
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"UsernameExistsException", "AliasExistsException"}:
            raise HttpError(409, "user with that email already exists") from exc
        raise

    cognito_user = created.get("User", {}) or {}
    new_username = cognito_user.get("Username", email)

    if is_admin_raw:
        cognito.admin_add_user_to_group(
            UserPoolId=settings.cognito_user_pool_id,
            Username=new_username,
            GroupName=ADMIN_GROUP,
        )

    logger.info(
        "admin_user_invited",
        extra={
            "actorUserId": actor.sub,
            "targetUsername": new_username,
            "email": email,
            "isAdmin": is_admin_raw,
        },
    )

    return ok(
        _user_response(cognito_user, is_admin=is_admin_raw),
        status=201,
    )


def _update_user(
    event: dict[str, Any],
    actor: AuthenticatedUser,
    target_username: str,
) -> dict[str, Any]:
    """Toggle enabled and/or admin-group membership for ``target_username``."""
    if not target_username:
        raise HttpError(400, "username is required")

    body = parse_json_body(event)
    enabled_change = body.get("enabled")
    is_admin_change = body.get("isAdmin")

    if enabled_change is not None and not isinstance(enabled_change, bool):
        raise HttpError(400, "enabled must be a boolean")
    if is_admin_change is not None and not isinstance(is_admin_change, bool):
        raise HttpError(400, "isAdmin must be a boolean")

    # Self-protection — fail BEFORE any mutation. Order matters: SelfDemote
    # gets first dibs over LastAdmin so the caller sees the more specific
    # message even when they're also the last admin.
    if (
        enabled_change is False
        and target_username == actor.sub
    ):
        raise HttpError(
            403, "you cannot disable your own account", error_type="SelfDisable"
        )
    if (
        is_admin_change is False
        and target_username == actor.sub
    ):
        raise HttpError(
            403,
            "you cannot remove yourself from the admins group",
            error_type="SelfDemote",
        )

    # Look up the user first so a 404 happens before we touch anything.
    current = _admin_get_user(target_username)
    current_enabled = bool(current.get("Enabled", False))

    settings = _settings()
    cognito = _cognito()

    changes: dict[str, Any] = {}

    # ---- enabled flip --------------------------------------------------
    if enabled_change is not None and enabled_change != current_enabled:
        if enabled_change:
            cognito.admin_enable_user(
                UserPoolId=settings.cognito_user_pool_id,
                Username=target_username,
            )
        else:
            cognito.admin_disable_user(
                UserPoolId=settings.cognito_user_pool_id,
                Username=target_username,
            )
            # Kill any cached refresh tokens so the disabled user can't keep
            # using the app for the next ~1 day.
            cognito.admin_user_global_sign_out(
                UserPoolId=settings.cognito_user_pool_id,
                Username=target_username,
            )
        changes["enabled"] = enabled_change

    # ---- admin group toggle -------------------------------------------
    is_admin_now = False
    if is_admin_change is not None:
        admin_set = _list_admin_usernames()
        currently_admin = target_username in admin_set
        if is_admin_change and not currently_admin:
            cognito.admin_add_user_to_group(
                UserPoolId=settings.cognito_user_pool_id,
                Username=target_username,
                GroupName=ADMIN_GROUP,
            )
            changes["isAdmin"] = True
            is_admin_now = True
        elif not is_admin_change and currently_admin:
            # LastAdmin guard — refuse if removing this user would empty
            # the admins group. Self-demote was already filtered above.
            if len(admin_set) <= 1:
                raise HttpError(
                    403,
                    "cannot remove the last admin from the admins group",
                    error_type="LastAdmin",
                )
            cognito.admin_remove_user_from_group(
                UserPoolId=settings.cognito_user_pool_id,
                Username=target_username,
                GroupName=ADMIN_GROUP,
            )
            changes["isAdmin"] = False
            is_admin_now = False
        else:
            is_admin_now = currently_admin
    else:
        # No group change requested — still need to compute current state
        # for the response shape. Cheaper than re-listing the whole group.
        is_admin_now = target_username in _list_admin_usernames()

    refreshed = _admin_get_user(target_username)
    response = _user_response(refreshed, is_admin=is_admin_now)

    logger.info(
        "admin_user_updated",
        extra={
            "actorUserId": actor.sub,
            "targetUsername": target_username,
            "changes": changes,
        },
    )

    return ok(response)


def _sign_out_user(
    actor: AuthenticatedUser, target_username: str
) -> dict[str, Any]:
    """Force-revoke all active sessions for ``target_username``."""
    if not target_username:
        raise HttpError(400, "username is required")

    settings = _settings()
    cognito = _cognito()

    try:
        cognito.admin_user_global_sign_out(
            UserPoolId=settings.cognito_user_pool_id,
            Username=target_username,
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code == "UserNotFoundException":
            raise HttpError(404, "user not found") from exc
        raise

    logger.info(
        "admin_user_signed_out",
        extra={
            "actorUserId": actor.sub,
            "targetUsername": target_username,
        },
    )
    return ok({"username": target_username, "signedOut": True})

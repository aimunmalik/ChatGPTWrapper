# Admin user management contract

> Single source of truth for the admin panel feature. Backend, frontend,
> and infra agents all build to this spec. If reality and this doc diverge,
> update this doc — don't drift.

---

## What it is

Admin-only UI in Praxis for managing Cognito users without dropping into
the AWS console. Covers the 80% of day-to-day user-management actions:
invite, promote/demote admin, enable/disable, force sign-out.

---

## Scope decisions (locked)

**In:** list, invite, toggle admin role, enable/disable, force sign-out
**Out (deferred):** delete, password reset, per-user audit log, bulk ops

The Praxis admin panel is a **secondary control**. With M365 SSO active,
the source of truth for "who works at ANNA" is M365 — disabling there
already cuts Praxis access within ~24h via our 1-day refresh-token TTL.
This panel is for: pre-existing Cognito users (break-glass), promoting
people to admin, and faster-than-M365 revocation when needed.

---

## API surface

All routes live under `/admin/users`. Every route requires
`http.require_admin(user)`.

### `GET /admin/users`

List every user in the pool. Defensively capped at 200 results — if you
hit that you've outgrown the simple Scan and need pagination (not in
MVP).

**Response 200:**
```json
{
  "users": [
    {
      "username": "6408a4e8-0091-709f-af5e-3e23f6193be7",
      "email": "aimun@annaautismcare.com",
      "name": "Aimun Malik",
      "status": "CONFIRMED",
      "enabled": true,
      "isAdmin": true,
      "identitySource": "local",
      "createdAt": 1745728800000,
      "lastModifiedAt": 1745782800000
    },
    {
      "username": "microsoft_8a9b1c2d3e4f5g6h7i8j9k0l",
      "email": "clinician@annaautismcare.com",
      "name": "Some Clinician",
      "status": "EXTERNAL_PROVIDER",
      "enabled": true,
      "isAdmin": false,
      "identitySource": "microsoft",
      "createdAt": 1745800000000,
      "lastModifiedAt": 1745800000000
    }
  ]
}
```

`identitySource` is derived from the username prefix: `microsoft_…` →
`"microsoft"`, otherwise `"local"`. We do NOT call separate APIs to look
up identity providers per user.

### `POST /admin/users`

Invite a new user by email. Cognito sends a welcome email with a
temporary password; on first sign-in the user sets a permanent password
and enrolls TOTP (existing pool config).

**Request body:**
```json
{
  "email": "newperson@annaautismcare.com",
  "name": "New Person",
  "isAdmin": false
}
```

**Response 201:** the new user object (same shape as list response).

**Errors:**
- `400` — email/name missing or malformed; isAdmin not boolean
- `409` — a user with that email already exists in the pool

### `PATCH /admin/users/{username}`

Update the user's enabled-state and/or admin-group membership. Both
fields are optional — sending only what you want to change. Idempotent.

**Request body:**
```json
{
  "enabled": false,
  "isAdmin": true
}
```

**Response 200:** updated user object.

**Errors:**
- `404` — username not found
- `403 SelfDisable` — caller cannot disable themselves
- `403 SelfDemote` — caller cannot demote themselves from admins
- `403 LastAdmin` — refusing to remove the last admin from the group

When `enabled` flips from `true → false`, the handler ALSO calls
`AdminUserGlobalSignOut` so the disabled user can't keep using a cached
session via their refresh token.

### `POST /admin/users/{username}/sign-out`

Force-revoke all active sessions for the user. Useful when you want to
invalidate cached tokens without disabling the account.

**Response 200:** `{ "username": "...", "signedOut": true }`

**Errors:**
- `404` — username not found

---

## Backend (`anna_chat/handlers/admin_users.py`)

Single handler module dispatching all four routes via the
`event["routeKey"]` pattern (mirror `handlers/kb.py`).

Key Cognito calls:
- `cognito.list_users` (paginated; cap at 200)
- `cognito.list_users_in_group(group_name="admins")` to compute `isAdmin`
  efficiently in one call instead of N queries (one per user)
- `cognito.admin_create_user` with `MessageAction="SUPPRESS"` if you
  want quiet creation, OR omit to let Cognito send the invite email
  (we want the email — keep default behavior)
- `cognito.admin_add_user_to_group` / `admin_remove_user_from_group`
- `cognito.admin_enable_user` / `admin_disable_user`
- `cognito.admin_user_global_sign_out`

**Self-protection logic:**
- `SelfDisable`: if `target_username == authenticated_user.sub` and
  request sets `enabled=false`, raise.
- `SelfDemote`: same but for removing self from admins.
- `LastAdmin`: when removing someone (anyone) from admins, query
  `list_users_in_group(admins)` first and refuse if they would be the
  last member.

**Returned user shape:** convert Cognito's `Attributes` array (list of
`{Name, Value}` dicts) into a flat `email` + `name`. Cognito's `Status`
field is what we expose as `status`; `Enabled` becomes `enabled`.

**For federated users** (`microsoft_…`): `Status` is `EXTERNAL_PROVIDER`.
Their email may live under the `email` attribute or under a custom
attribute depending on attribute-mapping; treat missing email as `""`.

Settings: add `cognito_user_pool_id` to `Settings` from the existing
`COGNITO_USER_POOL_ID` env var (already populated for all Lambdas via
`local.lambda_env`).

**Audit logging.** Every admin action logs:
```
{
  "event": "admin_user_<action>",       // invited, updated, signed_out
  "actorUserId": "<sub>",
  "targetUsername": "<username>",
  "changes": { "enabled": false, "isAdmin": true }   // for updates only
}
```
NEVER log the temp password Cognito returns (it doesn't echo it back to
us anyway, but if it ever does, redact). Email IS logged because it's
already in audit-class data per CloudTrail.

---

## Frontend (`frontend/src/components/AdminUsers.tsx`)

Admin-only modal accessed from the command palette ("Manage users…").
Same admin gate that hides "Manage knowledge base…" — `useIsAdmin()`.

Layout:

```
┌─ Manage users ──────────────────────────────[×]
│
│  + Invite a user
│  ┌──────────────────────┐ ┌──────────────┐
│  │ name@example.com     │ │ Display name │  ☐ Admin   [Send invite]
│  └──────────────────────┘ └──────────────┘
│
│  ─── 4 users ─────────────────────────────────
│
│  Aimun Malik       aimun@…    Local      🛡 Admin   ●          ⋯
│  Break Glass       aimunm83…  Local           ●          ⋯
│  Some Clinician    @anna…     Microsoft       ●          ⋯
│  Old Account       legacy@…   Local      🛡 Admin   ⊘ Disabled  ⋯
│
└──────────────────────────────────────────────
```

The `⋯` overflow per row opens a small popover:
- ☐ / ☑ Admin  (toggle group membership)
- Disable / Enable
- Sign out everywhere
- (Greyed out + tooltip when an action would self-block per backend rules)

**Optimistic updates** with rollback on error (mirror `useKbDocuments`).

**Inviting** shows the new user in the list immediately on `201`. The
invite email is async on Cognito's side; UI just confirms with a
toast/inline note: *"Welcome email sent to newperson@anna…"*

**File layout:**
- `src/api/admin.ts` — typed client
- `src/hooks/useAdminUsers.ts` — list + mutations + optimistic state
- `src/components/AdminUsers.tsx` — the modal
- `src/pages/ChatPage.tsx` — admin-gated command palette entry, mirroring `Manage knowledge base…`
- `src/styles/global.css` — append BEM blocks `.admin-users`, reuse `.cmdk-overlay` / `.cmdk-panel`

---

## Infra

### VPC interface endpoint

The new Lambda needs to call `cognito-idp.us-east-1.amazonaws.com`. Add
`"cognito-idp"` to `local.interface_endpoint_services` in
`infra/modules/network/main.tf`. Same fix-pattern as Textract and the
recent `lambda` endpoint.

### New Lambda: `lambda_admin`

In `infra/envs/dev/backend_compute.tf`:

```hcl
module "lambda_admin" {
  source          = "../../modules/lambda"
  function_name   = "anna-chat-${var.env}-admin"
  handler         = "anna_chat.handlers.admin_users.handler"
  zip_path        = local.lambda_zip_path
  timeout_seconds = 15
  memory_mb       = 512

  environment_variables = merge(local.lambda_env, {
    AWS_LAMBDA_LOG_FORMAT = "JSON"
  })

  log_retention_days = var.log_retention_days
  logs_kms_key_arn   = module.kms_logs.key_arn

  vpc_id         = module.network.vpc_id
  vpc_cidr       = module.network.vpc_cidr
  vpc_subnet_ids = module.network.private_subnet_ids

  cognito_user_pool_arn = module.cognito.user_pool_arn

  tags = local.tags
}
```

### IAM additions to `modules/lambda`

Add a new variable `cognito_user_pool_arn = ""` and a conditional inline
IAM statement. Action set:

```
cognito-idp:ListUsers
cognito-idp:ListUsersInGroup
cognito-idp:AdminGetUser
cognito-idp:AdminCreateUser
cognito-idp:AdminAddUserToGroup
cognito-idp:AdminRemoveUserFromGroup
cognito-idp:AdminEnableUser
cognito-idp:AdminDisableUser
cognito-idp:AdminUserGlobalSignOut
cognito-idp:AdminListGroupsForUser
```

Resource scoped to the user pool ARN. NEVER `Resource: "*"`.

### Cognito module output

`infra/modules/cognito/outputs.tf` already exposes `user_pool_id`. Add
`user_pool_arn` if it's not already there (most modules already include it).

### API Gateway routes

In `infra/envs/dev/backend_compute.tf` `module "api"` `routes`:

- `"GET /admin/users"` → `lambda_admin`
- `"POST /admin/users"` → `lambda_admin`
- `"PATCH /admin/users/{username}"` → `lambda_admin`
- `"POST /admin/users/{username}/sign-out"` → `lambda_admin`

---

## Tests

Backend test file: `backend/tests/test_admin_users.py`.

Use `botocore.stub.Stubber` against a real `cognito-idp` boto3 client
the handler module exposes for injection. Test cases:

- list returns merged user + admin group state
- invite happy path (creates user, optionally adds to admins, returns shape)
- invite duplicate email → 409
- update enable/disable round trip
- update group toggle round trip
- self-disable refused (403, errorType `SelfDisable`)
- self-demote refused (403, errorType `SelfDemote`)
- last-admin removal refused (403, errorType `LastAdmin`)
- sign-out happy path
- sign-out on missing user → 404
- non-admin caller blocked at handler entry (existing `require_admin` gate)

Aim for ~12-15 tests, on par with `test_kb.py` density.

---

## Out-of-scope explicit

- **Multi-tenant isolation** — single Cognito pool, no per-tenant slicing.
- **Custom user attributes / roles beyond `admins`** — V1 is just admin or not. Adding more groups (e.g. `clinicians`, `bcba`, `auditor`) is a future config.
- **Self-service signup** — explicitly not enabled (`allow_admin_create_user_only = true` already set on the pool).
- **Federated user provisioning workflow** — when someone first signs in via Microsoft, Cognito auto-creates the federated user. The admin panel will show them after that first sign-in. We do NOT pre-provision federated users.

# Personal prompt library contract (Phase 6d)

**Single source of truth** for the per-user prompt library. Backend, frontend, and Terraform work is scoped to this document.

## Goal

Each user has their own library of reusable prompt templates. They can create, edit, and delete their prompts. When they open the command palette, their prompts appear alongside the built-in BCBA templates — selecting either one starts a new chat with the template text pre-filled (same flow as today).

## User flow

1. User presses ⌘K (or clicks a new **"Prompts"** entry point in the header).
2. Command palette now shows TWO groups:
   - **My prompts** — user's saved templates
   - **Quick prompts** — the 11 hardcoded BCBA starters (unchanged)
3. Selecting either one inserts the template into the composer and starts a new chat (current behavior).
4. A new command **"Manage my prompts…"** (always in the palette) opens a Prompt Library modal. The modal has:
   - A list of the user's prompts (title + 1-line preview)
   - **New prompt** button → form with `title`, `body` fields
   - Edit pencil on each prompt
   - Delete × on each prompt
   - **Copy from built-in** — a secondary list of the 11 starters with a "Copy to my prompts" button so users can customize a starter

## Scope for this phase

**In:**
- Per-user CRUD of prompts
- DDB storage
- Command-palette integration (merge built-in + user)
- Library modal with basic form

**Out (defer to later phase if ever):**
- Team sharing / org-wide prompts
- Prompt folders or tags
- Import / export
- "Save this message as a prompt" from chat history
- Version history on edits

## DynamoDB schema

### New table: `anna-chat-{env}-prompts`

| Attribute | Type | Role |
|---|---|---|
| `userId` | S (PK) | Cognito sub |
| `promptId` | S (SK) | `p_` + 16 hex chars |
| `title` | S | Short name, ≤120 chars |
| `body` | S | Template text, ≤20000 chars |
| `createdAt` | N | Epoch ms |
| `updatedAt` | N | Epoch ms |

No GSI, no TTL. Prompts live forever until the user deletes them (or their entire userId is purged).

CMK-encrypted via `module.kms_dynamodb.key_arn`. PITR enabled. Deletion protection follows env (prod only).

## API

All routes authenticated via the existing Cognito JWT authorizer on the HTTP API. New Lambda `anna-chat-{env}-prompts` handles all four routes.

### `GET /prompts`

Response:
```json
{
  "prompts": [
    {
      "promptId": "p_abc123...",
      "title": "Draft BIP for aggressive behaviors",
      "body": "Draft a Behavior Intervention Plan for...\n\nClient: [...]\n",
      "createdAt": 1713000000000,
      "updatedAt": 1713100000000
    }
  ]
}
```

### `POST /prompts`

Request:
```json
{ "title": "Draft BIP for aggressive behaviors", "body": "Draft a..." }
```

Response:
```json
{ "prompt": {...full prompt object...} }
```

Validation:
- `title`: 1-120 chars, non-empty after trim
- `body`: 1-20000 chars, non-empty after trim

### `PUT /prompts/{promptId}`

Request: same shape as POST (both fields required — full replace semantics).

Response: updated prompt object. 404 if not found or not owned by caller.

### `DELETE /prompts/{promptId}`

Response: `{ "deleted": true }` or `{ "deleted": false }` if it didn't exist. 200/204.

## Backend

New files:
- `backend/src/anna_chat/prompts_repo.py` — DDB repository (`create`, `list_for_user`, `get`, `update`, `delete`)
- `backend/src/anna_chat/handlers/prompts.py` — HTTP handler with `routeKey` dispatch

Updates:
- `backend/src/anna_chat/settings.py` — new field `prompts_table` from `PROMPTS_TABLE` env var
- `backend/requirements.txt` — no new deps
- `backend/tests/test_prompts_repo.py` — new, Stubber-based

Handler rules (match the shape of other handlers):
- Use `authenticate(event, settings)` — 401 if missing
- Use `ok()` / `error()` / `HttpError` / `parse_json_body()` from `http.py`
- Log metadata only (promptId, userId, count, latencyMs) — **never the title or body**
- Ownership enforced by using `userId` as DDB PK on every operation

## Frontend

New files:
- `frontend/src/api/prompts.ts` — typed client: `listPrompts`, `createPrompt`, `updatePrompt`, `deletePrompt`
- `frontend/src/hooks/usePrompts.ts` — hook: `{ prompts, isLoading, error, create, update, remove, refresh }`
- `frontend/src/components/PromptLibrary.tsx` — modal with list + inline edit form

Updates:
- `frontend/src/pages/ChatPage.tsx` — instantiate `usePrompts()`; add a `promptLibraryOpen` state; inject user prompts into the command palette's "My prompts" group; add a `Manage my prompts…` command that opens the modal.
- `frontend/src/components/CommandPalette.tsx` — no change to component; commands are grouped by the `group` property already.
- `frontend/src/styles/global.css` — add `.prompt-library` styles (modal overlay reusing `.cmdk-overlay` pattern, form inputs matching the existing brand palette).

UI details:
- Modal overlay matches the command palette's look (blur + centered panel).
- Form: title input (single line) + body textarea (6 rows min, expands). "Save" and "Cancel" buttons. Body can be edited with the same shortcuts as the chat composer.
- Empty state: "You haven't saved any prompts yet. Starter templates are available under Quick prompts in ⌘K."
- "Copy from built-in" is a secondary tab/section in the modal — lists the 11 built-ins with a small copy-icon button per row that prefills the new-prompt form with the template body.

## Infra

New files:
- `infra/modules/prompts/main.tf` — DDB table (`anna-chat-{env}-prompts`), SSE-KMS with `var.kms_key_arn`, PITR on, deletion protection via `var.deletion_protection`.
- `infra/modules/prompts/variables.tf`
- `infra/modules/prompts/outputs.tf`

Updates:
- `infra/envs/dev/backend_compute.tf`:
  - Add `module "prompts"` using the new module. Pass `kms_key_arn = module.kms_dynamodb.key_arn`.
  - Add `module "lambda_prompts"` (512MB, 15s, Python handler `anna_chat.handlers.prompts.handler`, VPC-attached, grant DDB RW on the prompts table, KMS decrypt on ddb CMK).
  - Update `local.lambda_env` to include `PROMPTS_TABLE = module.prompts.table_name`.
  - Add the four prompt routes to the `api` module's `routes` map (all → `lambda_prompts`).
- `infra/envs/dev/outputs.tf` — add `prompts_table_name`.

## Environment variables

New Lambda env:

| Name | Value |
|---|---|
| `PROMPTS_TABLE` | DDB table name from the prompts module |

Existing Lambdas do NOT need this env var — only the prompts handler Lambda uses the table.

## Audit

- CloudWatch Logs: prompt events logged with `userId`, `promptId`, `action` (created/updated/deleted), `titleLen`, `bodyLen`, `latencyMs` — never title or body text.
- No CloudTrail data events required for this table in this phase.

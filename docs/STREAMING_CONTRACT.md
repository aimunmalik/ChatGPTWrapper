# Streaming Chat Contract (Path B — async worker + poll)

Single source of truth for the live-updating chat feature. Implemented this way
because **Lambda Function URL response streaming does not work in a VPC**
(confirmed: AWS "VPC compatibility with response streaming" docs) and our chat
Lambda is VPC-bound by HIPAA design. API Gateway response streaming is REST-API
only (Nov 2025) and would require a new REST API + Node rewrite. So we reuse the
proven async-worker + poll pattern (same shape as the translate jobs feature).

## Why
`POST /chat` is synchronous and bounded by API Gateway HTTP API's hard **30s**
integration cap. A long multi-part question on Opus exceeds that → `503`. The
streaming path moves generation into a background worker (15-min Lambda budget)
and the browser polls for the answer as it builds.

## Flow
```
Browser ── POST /chat/stream ──▶ chat Lambda (kickoff)
                                   • persist user message
                                   • create empty assistant message (status=streaming)
                                   • async-invoke chat_worker (InvocationType=Event)
                                   • return {conversationId, messageId, sortKey, status:"streaming"}
chat_worker (async, ≤15 min)       • rebuild history, KB retrieval
                                   • bedrock.invoke_stream(...)
                                   • every ~0.8s: update message content
                                   • finalize: content + sources + tokens + status=complete|error
Browser ── GET /chat/stream?cid=&sk= (poll ~800ms) ──▶ chat Lambda
                                   • return {status, content, sources, tokens}
                                   • stop when status in {complete, error}
```

## Routes (existing HTTP API, JWT authorizer — no new API, no Function URL)

### POST /chat/stream  (kickoff)
Request (same shape as POST /chat):
```json
{ "message": "string (required, ≤20000 chars)",
  "conversationId": "string (optional)",
  "model": "string (optional, must be in ALLOWED_MODELS)" }
```
Response `202`:
```json
{ "conversationId": "c_…", "messageId": "m_…",
  "sortKey": "0001719…#m_…", "status": "streaming" }
```

### GET /chat/stream?cid={conversationId}&sk={sortKey}  (poll)
`cid` and `sk` are **query string** params (sortKey contains `#`, so the client
must `encodeURIComponent` it). Response `200`:
```json
{ "conversationId": "c_…", "messageId": "m_…", "sortKey": "…",
  "status": "streaming" | "complete" | "error",
  "content": "partial or full assistant text",
  "sources": [ /* KB Source objects, populated when complete */ ],
  "tokens": { "input": 0, "output": 0 },
  "model": "us.anthropic.claude-…" }
```
Ownership: the conversation must belong to the caller (404 otherwise); the
message's userId must match (404 otherwise).

## Message status (messages table)
New `status` attribute on the Message row:
- `complete` — normal finished message; ALSO the default for every legacy row
  (the dataclass default fills in when the attribute is absent on read).
- `streaming` — worker still generating. Excluded from `recent_turns_for_model`
  so an in-flight placeholder never pollutes model history.
- `error` — generation failed; `content` holds whatever streamed before failure.

## Worker invoke payload (Event)
```json
{ "conversationId": "c_…", "userId": "<cognito sub>", "sortKey": "…#m_…",
  "messageId": "m_…", "userMessage": "raw user text", "model": "us.anthropic.…" }
```

## Backend modules (DONE)
- `bedrock_client.py`: `invoke_stream()` (yields `{type:delta,text}` … `{type:done,…}`);
  `STREAM_READ_TIMEOUT`; `max_attempts` ctor param (worker uses 1 — never retry a stream).
- `ddb.py`: `Message.status`; `create_streaming_message`, `update_streaming_content`,
  `finalize_message`, `get_message`; `recent_turns_for_model` excludes `streaming`.
- `handlers/chat_core.py`: shared SYSTEM_PROMPT, ALLOWED_MODELS, singletons,
  KB/knowledge/attachment helpers, `build_turn()`.
- `handlers/chat.py`: routeKey dispatch — `_sync_chat` (legacy), `_start_stream`, `_poll_stream`.
- `handlers/chat_worker.py`: the async streaming worker.
- `settings.py`: `chat_worker_function_name` (env `CHAT_WORKER_FUNCTION_NAME`).

## Infra (TODO — task 6)
- New Lambda `lambda_chat_worker` (handler `anna_chat.handlers.chat_worker.handler`),
  VPC-resident, **timeout 900s, memory 1024MB**. Same env + IAM as `lambda_chat`
  (Bedrock invoke incl. Titan embeddings, DDB RW conversations+messages, DDB read
  kb + attachments). Mirror `lambda_translate_worker` for the module shape.
- On `lambda_chat` (the chat handler): add env `CHAT_WORKER_FUNCTION_NAME =
  module.lambda_chat_worker.function_name`, and IAM `lambda:InvokeFunction` on the
  worker ARN (mirror how the translate handler is granted invoke on its worker).
- Two new routes on the existing HTTP API → `lambda_chat`:
  `POST /chat/stream`, `GET /chat/stream`.
- The `lambda` VPC interface endpoint already exists (translate uses it). No new endpoint.

## Frontend (TODO — task 7)
- `api/chat.ts`: `postChatStream()` (POST /chat/stream) + `pollChatStream(cid, sk)`
  (GET with encoded query params) + types.
- `ChatView.tsx`: `handleSubmit` kicks off then polls ~800ms, appending `content`
  to the pending assistant draft until `status` is terminal; on `complete` set
  `sources`; on `error` show the partial + an error note. Stop polling on
  conversation switch / unmount and after a safety cap (~16 min).

## PHI logging
Never log message content, chunk text, or KB material. IDs, counts, token
totals, model id, and error **types** only — same rule as translate_worker.

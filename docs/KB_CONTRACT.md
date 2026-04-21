# Knowledge Base / RAG contract (Phase 7)

**Single source of truth** for retrieval-augmented generation. Backend, frontend, and Terraform work scoped to this document.

## Goal

Admin users (ANNA Cognito `admins` group) upload reference documents — clinical protocols, research articles, initial training, parent-training modules. Uploaded docs are chunked, embedded via Bedrock Titan Text Embeddings v2, and indexed in DynamoDB. At chat time, the user's message is embedded and the top-K most relevant chunks are injected into the prompt. Claude's answer cites the sources used, and those citations are rendered in the chat UI.

## What is and isn't PHI

- **KB content** is **not PHI** — protocols, research, training material is published / internal reference material, never patient-specific.
- **User queries** at chat time ARE PHI-possible (e.g., "what's our protocol for aggression in a 7-year-old during transitions"). The retrieval path must never log the query text.
- Same CMK / VPC / HIPAA-covered AWS posture as the rest of the stack applies.

## DynamoDB schema

### New table: `anna-chat-{env}-kb`

Single table stores both **document metadata** and **chunks**, differentiated by sort key prefix:

| Attribute | Type | Role |
|---|---|---|
| `kbDocId` | S (PK) | `kd_` + 16 hex — one per uploaded document |
| `sk` | S (SK) | `"META"` for doc metadata; `"chunk#0001"`, `"chunk#0002"`, ... for chunks (zero-padded to 4 digits, supports up to 9999 chunks/doc) |

**META items** (one per doc):

| Attribute | Notes |
|---|---|
| `docTitle` | S — user-supplied or derived from filename |
| `sourceType` | S — one of: `research` / `training` / `protocol` / `parent-training` / `other` |
| `filename` | S |
| `contentType` | S (`application/pdf` or `application/vnd.openxmlformats-officedocument.wordprocessingml.document`) |
| `sizeBytes` | N |
| `s3Key` | S (`kb/{kbDocId}/{sanitizedFilename}`) |
| `status` | S — `uploading` → `extracting` → `chunking` → `embedding` → `ready` / `error` |
| `statusMessage` | S — class name on error (no traceback) |
| `totalChunks` | N |
| `uploadedBy` | S — Cognito sub of the admin who uploaded |
| `tags` | SS — optional string set for later filtering |
| `createdAt` | N |
| `updatedAt` | N |

**Chunk items:**

| Attribute | Notes |
|---|---|
| `docTitle` | S — denormalized so retrieval doesn't need a second lookup |
| `sourceType` | S — same |
| `chunkIdx` | N — zero-based index |
| `chunkText` | S — up to ~4 KB text |
| `chunkTokens` | N — approximate token count |
| `embedding` | L[N] — 1024-dim float vector from Titan Embed v2 |
| `pageNumber` | N — optional, for PDFs |
| `sectionTitle` | S — optional |
| `createdAt` | N |

Encryption: CMK via `module.kms_dynamodb.key_arn`. PITR on. `deletion_protection` toggled by env (prod true, dev false).

No GSI for MVP — retrieval uses `Scan` with filter expression `begins_with(sk, :chunk_prefix)`. Fine at < 5K chunks, will need re-index to an OpenSearch-style store beyond that.

## S3 bucket: `anna-chat-{env}-kb-{account}`

- SSE-KMS with `module.kms_s3.key_arn`
- Versioning on
- Public access block (all 4)
- TLS-only bucket policy
- Lifecycle: current versions expire after 365 days (way longer than attachments — this is reference material); noncurrent after 60 days
- **No Object Lock** (different from attachments — this is editable admin content, not audit-immutable)
- CORS: `PUT, POST` from local dev + CloudFront + praxis custom domain
- S3 event notification: `ObjectCreated:*` under `kb/` prefix → ingestion Lambda

Object key: `kb/{kbDocId}/{sanitizedFilename}` — sanitize: `[^A-Za-z0-9._-]` → `_`, cap at 128 chars.

## Embeddings

- **Model**: `amazon.titan-embed-text-v2:0` (1024 dims, cosine similarity, HIPAA-eligible via AWS BAA)
- **Chunking**: 800 tokens with 100-token overlap. Rough token count via `len(text.split())` × 1.3 (good enough for chunking decisions — not accurate enough for billing).
- **Extraction**: reuse `backend/src/anna_chat/extractors.py`:
  - PDF: `_extract_textract` (existing Textract path)
  - DOCX: `_extract_docx` (existing python-docx path)
  - TXT: `_extract_txt`
  - CSV: `_extract_csv`
  - XLSX / images: **not supported for KB in MVP** — reject at upload with a clear error. Images and spreadsheets don't RAG well as plain text.

**User must enable Titan Embeddings v2 in the Bedrock console** — same "Model access" request flow as Claude. If access isn't granted, ingestion returns a clear error.

## Retrieval

Called synchronously from `chat.handler` before `bedrock.invoke`:

```python
def retrieve(query: str, *, top_k: int = 5, min_score: float = 0.35) -> list[RetrievedChunk]:
    query_vec = bedrock.embed(query)           # ~50 ms
    all_chunks = kb_repo.scan_chunks()         # cached at Lambda cold start; ~200 ms
    scored = sorted(
        ((cosine(query_vec, c.embedding), c) for c in all_chunks),
        reverse=True,
    )
    return [c for score, c in scored[:top_k] if score >= min_score]
```

**Performance budget**: at 1000 chunks × 1024 floats × 4 bytes = 4 MB cached in memory per Lambda container. Cold start pulls from DDB once, then serves from memory for ~20 min of inactivity. Cosine similarity over 1000 vectors ≈ 50 ms. Total retrieval overhead per chat turn: ~100 ms warm, ~500 ms cold.

**Thresholds** (tune later):
- `top_k = 5` chunks max
- `min_score = 0.35` (under the threshold = no relevant content; return empty list)
- If zero chunks pass, the prompt says `<knowledge>No relevant material found in the ANNA knowledge base.</knowledge>` — Claude answers from general knowledge without pretending to cite.

**Prompt injection format:**

```
<knowledge>
[Source 1] ANNA Training Module 3 — section: Transitions (page 12)
{chunk text}

---

[Source 2] Parent Training Handbook v2 — section: Managing Meltdowns
{chunk text}
</knowledge>

{user's actual message}
```

The system prompt gains:

> When the `<knowledge>` block contains relevant material, prefer it over your general knowledge and cite the source number in your response like `[1]` or `[Source 2]`. When no relevant material is returned, answer from general knowledge and say so briefly.

## Chat response shape (extended)

`POST /chat` response now includes `sources`:

```json
{
  "conversationId": "...",
  "messageId": "...",
  "assistantMessage": "Based on the ANNA transition protocol [1], you should...",
  "tokens": {"input": 812, "output": 204},
  "model": "us.anthropic.claude-sonnet-4-6",
  "sources": [
    { "index": 1, "docTitle": "ANNA Training Module 3", "sourceType": "training", "pageNumber": 12, "score": 0.82 }
  ]
}
```

`sources` is `[]` when no chunks were retrieved. The assistant may reference `[1]`, `[2]`, etc., which the frontend links to the corresponding entry in `sources`.

Persist `sources` with the assistant message in the messages table so conversation history renders citations on reload.

## API

All routes authenticated via the existing Cognito JWT authorizer. **Admin-only** routes additionally check that the authenticated user is in the `admins` Cognito group (via `cognito:groups` claim) — return 403 if not.

### `POST /kb/presigned-upload` — admin only

Request:
```json
{
  "filename": "ANNA Training Module 3.pdf",
  "contentType": "application/pdf",
  "sizeBytes": 2345678,
  "docTitle": "ANNA Training Module 3",
  "sourceType": "training"
}
```

Response:
```json
{
  "kbDocId": "kd_abc123...",
  "uploadUrl": "...",
  "uploadFields": {...},
  "expiresAt": 1713456789000
}
```

Validation:
- `sizeBytes` ≤ 100 MB (larger than attachments — clinical docs can be big)
- `contentType` ∈ {pdf, docx, txt, csv}
- `sourceType` ∈ allowed set
- `docTitle`: 1-200 chars
- Creates the META item with `status=uploading` and returns presigned POST scoped to `kb/{kbDocId}/`.

### `GET /kb/documents` — admin only

Response:
```json
{
  "documents": [
    {
      "kbDocId": "kd_...",
      "docTitle": "ANNA Training Module 3",
      "sourceType": "training",
      "filename": "training.pdf",
      "sizeBytes": 2345678,
      "status": "ready",
      "totalChunks": 47,
      "uploadedBy": "<cognito sub>",
      "createdAt": 1713000000000
    }
  ]
}
```

### `DELETE /kb/documents/{kbDocId}` — admin only

Deletes META row, all chunk rows (batch-write), and the S3 object. Returns `{"deleted": true}`.

## Backend

**New files:**
- `backend/src/anna_chat/kb_repo.py` — DDB repository: `create_doc`, `get_doc`, `list_docs`, `update_doc_status`, `delete_doc`, `write_chunks`, `scan_all_chunks`
- `backend/src/anna_chat/embeddings.py` — thin Bedrock Runtime wrapper around `amazon.titan-embed-text-v2:0`
- `backend/src/anna_chat/chunking.py` — sliding-window token-ish chunker (split on paragraphs, pack to ~800 tokens with 100-token overlap)
- `backend/src/anna_chat/kb_retrieve.py` — `retrieve(query, top_k, min_score)` — calls embeddings + kb_repo, cosine-sim ranks
- `backend/src/anna_chat/handlers/kb.py` — HTTP handler for the 3 admin routes
- `backend/src/anna_chat/handlers/kb_ingest.py` — S3 event handler: read → extract → chunk → embed → write, with status updates at each stage
- `backend/tests/test_kb_repo.py` — Stubber tests
- `backend/tests/test_chunking.py` — deterministic chunker tests
- `backend/tests/test_kb_retrieve.py` — cosine-sim correctness with synthetic vectors

**Modified:**
- `backend/src/anna_chat/handlers/chat.py`:
  - Call `kb_retrieve.retrieve()` before `bedrock.invoke`
  - Prepend `<knowledge>...</knowledge>` block to user_message
  - Update system prompt with citation instructions
  - Include `sources` array in response + persist on the assistant message
- `backend/src/anna_chat/settings.py`: add `kb_table`, `kb_bucket`, `kb_max_size_bytes` (100 MB default)
- `backend/src/anna_chat/handlers/conversations.py`: include `sources` on messages when loading a conversation
- `backend/src/anna_chat/http.py`: add `require_admin(user)` helper that raises `HttpError(403, "admin only")` if `"admins" not in user.groups`

**Admin check** — the existing `AuthenticatedUser` already parses `cognito:groups`. Just add:
```python
def require_admin(user: AuthenticatedUser) -> None:
    if "admins" not in user.groups:
        raise HttpError(403, "admin only")
```

## Frontend

**New files:**
- `frontend/src/api/kb.ts` — typed client: `uploadKbDoc`, `listKbDocs`, `deleteKbDoc`
- `frontend/src/components/KnowledgeBase.tsx` — modal (same `.cmdk-overlay` pattern as PromptLibrary):
  - Shows only to admins (the parent ChatPage passes `isAdmin` prop based on `useAuth().user.profile["cognito:groups"]`)
  - Upload form: file picker (accept PDF + DOCX + TXT + CSV), `docTitle`, `sourceType` dropdown
  - Uploaded-docs list: columns filename / title / type / status / uploaded / × delete
  - Polls `GET /kb/documents` every 2s while any doc has a non-terminal status
- `frontend/src/components/MessageSources.tsx` — below each assistant message, if `sources.length > 0`, render a small "Sources" row with numbered pills that link to each source (no external nav — just a hover tooltip showing the excerpt / doc title)

**Modified:**
- `frontend/src/pages/ChatPage.tsx`:
  - Add `kbOpen` state + a command palette entry `Library → Manage knowledge base…` (admin-only — hidden from non-admins)
  - Render `<KnowledgeBase>` modal
  - Pass `sources` through to `<ChatView>`, which passes to `<Message>`
- `frontend/src/components/ChatView.tsx` + `Message.tsx`:
  - Accept `sources: Source[]` on assistant messages
  - Render `<MessageSources>` below the body
- `frontend/src/api/chat.ts`: extend `ChatResponse` with `sources: Source[]` field
- `frontend/src/styles/global.css`: append `.message__sources`, `.message__source-pill` styles matching the attachment-chip pattern — pink-tinted, pill-shaped, subtle

**Admin check** — Cognito ID tokens include `cognito:groups` as a JSON array of strings. `useAuth().user?.profile["cognito:groups"]` returns that array. Wrap in a helper:

```ts
// frontend/src/auth/useIsAdmin.ts
export function useIsAdmin(): boolean {
  const { user } = useAuth();
  const groups = user?.profile["cognito:groups"] as string[] | undefined;
  return Array.isArray(groups) && groups.includes("admins");
}
```

## Infra

**New files:**
- `infra/modules/kb/main.tf` — S3 bucket (SSE-KMS, versioning, BPA, TLS-only, CORS for PUT/POST, lifecycle 365/60 days). DDB table (PK `kbDocId`, SK `sk`). S3 event notification → ingest Lambda.
- `infra/modules/kb/variables.tf`, `outputs.tf`

**Modified:**
- `infra/envs/dev/backend_compute.tf`:
  - New `module "kb"` (S3 + DDB table)
  - New `module "lambda_kb_ingest"` — 3072 MB / 600s timeout (embeddings + Textract can be slow), VPC-attached, grants: kb DDB RW, KB S3 GetObject, S3 CMK decrypt, DDB CMK, `bedrock:InvokeModel` on Titan embeddings, Textract
  - New `module "lambda_kb"` for admin API — 512 MB / 15s, grants: kb DDB RW, KB S3 PutObject + DeleteObject, KMS on both CMKs
  - Existing `module "lambda_chat"` gets `kms_key_arns` for the KB keys (same DDB CMK) + additional `bedrock_model_arns` for Titan embeddings + **access to the kb DDB table for retrieval**
  - 3 new API routes under the api module: `POST /kb/presigned-upload`, `GET /kb/documents`, `DELETE /kb/documents/{kbDocId}` → `lambda_kb`
  - `local.lambda_env` gains `KB_TABLE`, `KB_BUCKET`, `KB_MAX_SIZE_BYTES=104857600` (100 MB)
- `infra/envs/dev/outputs.tf`: `kb_bucket_name`, `kb_table_name`

**Titan model ARN for IAM:**
```
arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0
```

(Add to `local.bedrock_model_arns` or a new list.)

## Audit

CloudWatch logs — only metadata. **Never** log:
- Query text
- Chunk text
- Embedding values

Do log:
- `kbDocId`, `chunkIdx`, `totalChunks`, `tokensEmbedded`, `status`, `sourceType`, `scoreRange`, `chunksRetrieved`, `topScore`, `latencyMs`, `userId`

## Out of scope for Phase 7

- Hybrid keyword + semantic search
- Query rewriting / HyDE
- Per-user or per-group access control on specific docs (everything admin-uploaded is retrievable by everyone, as designed)
- Bulk folder / zip import
- Inline diff / versioning of edited protocols
- Non-text sources (images as OCR, audio transcripts, video)
- Re-embedding on model upgrade (we pin Titan v2; if we later move to v3 there's a migration)

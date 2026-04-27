# Translation jobs contract

> Single source of truth for the document-translation feature. Backend, frontend,
> and infra agents all build to this spec. If reality and this doc diverge,
> update this doc — don't drift.

---

## What it is

User has an attached document on a chat conversation. They want the document
translated into another language and delivered as `.docx` + `.pdf` downloads.

Translation can take 30s–5min depending on length, which is well outside
API Gateway's 30s integration timeout — so this is a **background job**,
not a synchronous request. UI submits a job, polls for status, downloads
the artifacts when ready.

---

## End-to-end flow

1. User attaches a document via the existing attachment flow (PDF, DOCX, TXT, CSV).
2. On the AttachmentChip's overflow menu, user clicks **"Translate…"** → opens TranslateDialog.
3. TranslateDialog shows the source filename + a target-language dropdown. User picks language → submits.
4. Frontend calls `POST /translate/jobs` with `{ attachmentId, targetLanguage }`.
5. Backend handler:
   - Authenticates user
   - Validates that the user owns this attachment
   - Creates a job row in DynamoDB with `status="pending"`
   - **Async-invokes** the worker Lambda (`lambda_translate_worker`) with `{ jobId }`
   - Returns `{ jobId, status: "pending" }` immediately
6. Frontend renders a **JobCard** in chat: *"Translating Crank et al 2021 to Spanish… ~2 minutes"*. Polls `GET /translate/jobs/{jobId}` every 5s.
7. Worker Lambda:
   - Reads source attachment text from S3 (already extracted by the existing attachment flow)
   - Updates job status to `extracting → chunking → translating → formatting → ready`
   - Translates via Bedrock in chunks of ≤4000 input tokens, concatenating outputs
   - Generates `.docx` (python-docx) and `.pdf` (reportlab)
   - Uploads both to S3 under `translations/{jobId}/{filename}.{docx,pdf}`
   - Updates DDB row with output keys + `status="ready"`
8. Frontend's poll sees `status="ready"` → JobCard shows two download buttons.
9. User clicks → frontend calls `GET /translate/jobs/{jobId}/download/{format}` → backend returns a 5-min presigned S3 GET URL → browser navigates → file downloads.

---

## DynamoDB schema

**Table:** `anna-chat-{env}-jobs`

| Field | Type | Notes |
|---|---|---|
| `userId` | S | Cognito sub. **Partition key.** |
| `jobId` | S | `tj_<16-hex-chars>` (translation job). **Sort key.** |
| `jobType` | S | `"translate"` (future-proof for other job types) |
| `status` | S | `pending`, `extracting`, `chunking`, `translating`, `formatting`, `ready`, `error` |
| `statusMessage` | S | Human-readable detail when `status=error`; otherwise `""` |
| `sourceAttachmentId` | S | The `att_*` ID from the attachments table |
| `sourceFilename` | S | Original filename for display |
| `sourceContentType` | S | MIME of the source |
| `targetLanguage` | S | ISO-639 code (`es`, `zh`, `vi`, etc.) — see § Languages |
| `targetLanguageLabel` | S | Display name (`"Spanish"`, `"Mandarin"`, etc.) |
| `outputDocxKey` | S | S3 key of the .docx output (only when `status=ready`) |
| `outputPdfKey` | S | S3 key of the .pdf output |
| `inputTokensApprox` | N | Approximate input token count, set during chunking |
| `outputTokensApprox` | N | Cumulative output tokens across all chunks |
| `createdAt` | N | epoch ms |
| `updatedAt` | N | epoch ms |
| `ttl` | N | epoch SECONDS, 7 days from createdAt — auto-expires DDB row |

PITR enabled. SSE-KMS using the existing `kms_dynamodb` key. PAY_PER_REQUEST.
Deletion protection on in prod, off in dev.

No GSI for V1 — listing jobs uses `Query` on partition key.

---

## S3 layout

Translations live in the existing `attachments` bucket under a dedicated prefix:

```
translations/
  tj_abc123def456ffff/
    Crank_et_al_2021_es.docx
    Crank_et_al_2021_es.pdf
```

Lifecycle: same as the rest of the bucket (90-day expiration, noncurrent versions 60d).

The translation Lambda gets `s3:PutObject` on this prefix and `s3:GetObject` on the existing attachment text prefix. The handler Lambda gets `s3:GetObject` on the translations prefix (to mint presigned URLs).

---

## API surface

All routes are JWT-authenticated via the existing API Gateway authorizer. None are admin-gated — every signed-in user can translate their own documents.

### `POST /translate/jobs`
Create a translation job.

**Request body:**
```json
{
  "attachmentId": "att_abc123",
  "targetLanguage": "es"
}
```

**Response 201:**
```json
{
  "jobId": "tj_abc123def456ffff",
  "status": "pending",
  "sourceFilename": "Crank et al 2021.pdf",
  "targetLanguageLabel": "Spanish",
  "createdAt": 1745782800000
}
```

**Errors:**
- `400` — attachmentId / targetLanguage missing or invalid; targetLanguage not in supported list
- `403` — attachment not owned by caller
- `404` — attachment not found OR not yet `status=ready` (no extracted text to translate)

### `GET /translate/jobs/{jobId}`
Poll status.

**Response 200:**
```json
{
  "jobId": "tj_abc123def456ffff",
  "userId": "<sub>",
  "status": "translating",
  "statusMessage": null,
  "sourceFilename": "Crank et al 2021.pdf",
  "targetLanguage": "es",
  "targetLanguageLabel": "Spanish",
  "createdAt": 1745782800000,
  "updatedAt": 1745782815000,
  "downloadDocxUrl": null,
  "downloadPdfUrl": null
}
```

When `status="ready"`, `downloadDocxUrl` and `downloadPdfUrl` are absolute paths
to the download endpoints below (NOT presigned URLs — those expire too fast for
a polling cycle):
```
"downloadDocxUrl": "/translate/jobs/tj_abc/download/docx",
"downloadPdfUrl":  "/translate/jobs/tj_abc/download/pdf"
```

**Errors:**
- `404` — jobId not found OR not owned by caller (deliberately collapsed to avoid existence enumeration)

### `GET /translate/jobs/{jobId}/download/{format}`
Returns a 5-minute presigned GET URL for the file.

`{format}` ∈ `{"docx", "pdf"}`.

**Response 200:**
```json
{
  "url": "https://...s3.amazonaws.com/...?X-Amz-Algorithm=...",
  "expiresAt": 1745782815000,
  "filename": "Crank_et_al_2021_es.docx",
  "contentType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
}
```

Frontend calls this on click, then navigates `window.open(url)` to trigger download.

**Errors:**
- `400` — invalid format
- `404` — jobId not found / not owned / not yet ready

### `GET /translate/jobs`
List the caller's recent translation jobs (newest first, last 50).

**Response 200:**
```json
{
  "jobs": [
    { /* same shape as GET /translate/jobs/{jobId} */ },
    ...
  ]
}
```

Used by the "Recent translations" panel in the command palette / sidebar.

---

## Worker Lambda (`lambda_translate_worker`)

Invoked **asynchronously** (`InvocationType=Event`) by the handler. Payload:
```json
{ "jobId": "tj_abc", "userId": "<sub>" }
```

State machine:
```
pending → extracting → chunking → translating → formatting → ready
                                                            \→ error (any stage)
```

Each transition writes `updatedAt`. If anything raises, status flips to `error`
with `statusMessage = type(exc).__name__`. Specific error codes for known cases:
- `SourceNotReady` — attachment isn't extracted yet
- `SourceNotFound` — attachment row missing or S3 object gone
- `TooLarge` — extracted text exceeds 200K tokens (Bedrock context limit) — V1 doesn't split-and-merge across separate model calls beyond chunked translation
- `BedrockThrottled` — exceeded retries
- `UnsupportedLanguage` — target language not in supported list

Lambda config:
- Runtime: Python 3.12
- Memory: 2048 MB
- Timeout: **900s (15 min, the AWS hard cap)**
- VPC-resident (same private subnets as everything else)
- IAM: bedrock invoke (Sonnet 4.6 + Haiku 4.5), s3 read/write attachments bucket, ddb r/w on jobs table, kms decrypt on attachment + s3 keys
- ENV: `JOBS_TABLE`, `ATTACHMENTS_TABLE`, `ATTACHMENTS_BUCKET`, `BEDROCK_MODEL_ID`

---

## Translation logic

Source text comes from the attachment's already-extracted text in DynamoDB
(`extractedText` field on the attachment row). No re-extraction.

**Chunking:** Split paragraph-aware into windows of ≤3000 input tokens (rough
heuristic: `len(text.split()) * 1.3`). 200-token overlap between windows so
sentence/paragraph boundaries that fall mid-chunk get translated coherently.

**Per-chunk Bedrock call:** uses `BEDROCK_MODEL_ID` (default Sonnet 4.6),
`max_tokens=4096`, system prompt:

```
You are a professional medical translator. Translate the following text to {target_language_label}.
Preserve paragraph breaks, lists, headings, and any formatting cues. Do not add commentary,
introductions, or notes — output only the translation. Maintain person-first language.
```

User message is the chunk text verbatim.

**Concatenation:** Outputs joined with `\n\n` between chunks. Overlap regions
get de-duplicated by simple line-prefix matching (best-effort — perfect
deduplication isn't worth the complexity).

**Token budgeting:** if `inputTokensApprox > 200_000`, fail with
`TooLarge` rather than attempting (Sonnet's context window).

---

## Formatters

### `.docx` (python-docx, already a dep)

- Default Calibri 11pt body, default heading styles
- Title at top: original filename, target language, "Translated by Praxis"
- Body: paragraphs preserved (split on blank lines), no fancy markdown rendering for V1 — translation is plain text in, plain text out
- Right-to-left script handling (Arabic, Hebrew): set paragraph `paragraph_format.rtl = True` when target language is RTL

### `.pdf` (reportlab — NEW dep, `reportlab==4.2.5`)

- Letter size, 1" margins
- Helvetica 11pt body, Helvetica-Bold 14pt for the header
- Header: same title block as .docx
- Body: paragraphs, simple flow
- For V1, no Unicode complications: stick with the default fonts. If the target language requires extended scripts (Arabic, Devanagari, Cantonese, etc.) and the default font can't render them, fall back to using a Unicode-safe font like `STIXTwoText` if shipped with reportlab. Document this as a known V1 limitation if it bites.

---

## Languages

V1 supported list — common languages for ANNA's clinical population:

| ISO | Label | RTL |
|---|---|---|
| `es` | Spanish | no |
| `zh` | Mandarin (Simplified Chinese) | no |
| `vi` | Vietnamese | no |
| `tl` | Tagalog | no |
| `ko` | Korean | no |
| `ru` | Russian | no |
| `ar` | Arabic | **yes** |
| `fr` | French | no |
| `pt` | Portuguese (Brazilian) | no |
| `hi` | Hindi | no |
| `de` | German | no |
| `ja` | Japanese | no |
| `en` | English (e.g. translate non-English source → English) | no |

Backend rejects anything outside this set. Adding a language is a one-line
config change in `LANGUAGES` constant in the worker.

---

## UI surface

### AttachmentChip "Translate…" menu

The existing `AttachmentChip` gets an overflow `⋯` button with a popover.
First menu item: **"Translate…"**. Click → opens `TranslateDialog`.

### TranslateDialog

Modal (reuse the `cmdk-overlay` + `cmdk-panel` shell).

- Title: "Translate document"
- Read-only field: source filename
- Dropdown: target language
- Submit button: "Start translation"
- Below submit: link "View past translations →" → opens JobsList

### JobCard in chat

When a translation job is created, a card appears in the chat thread (above
the composer, like a system message):

```
🌐  Translating "Crank et al 2021.pdf" to Spanish
    Status: Translating page 3 of 8…   [Cancel]
```

Polls `GET /translate/jobs/{jobId}` every 5s while status is non-terminal.
On `status=ready` flips to:
```
🌐  Translation ready: "Crank et al 2021 (Spanish)"
    [Download .docx]   [Download .pdf]
```

On `status=error` flips to a red error card with the `statusMessage`.

JobCards persist in the chat session's local state (not on the server-side
message timeline) — refreshing the page loses them. The "Recent translations"
panel is the persistent record.

### Recent translations panel

⌘K → "Recent translations" command. Opens a modal listing the last 50 jobs
with status pills + download buttons. Same visual language as
`KnowledgeBase.tsx`'s upload list.

---

## What's deliberately out of scope for V1

- **Translating attachments not yet `status=ready`** — error fast, ask user to wait for extraction.
- **Cancel mid-translation** — the Cancel button on the JobCard just removes the card from local state; the worker keeps running. Real cancellation would need a Stop signal mechanism.
- **Custom translation glossaries** — stick with general medical translation prompt.
- **Side-by-side diff view** — output is just the translation, not source-aligned.
- **Multi-file batch** — one job, one source doc. Batch can be a wrapper later.
- **Translating directly from chat ("translate this for me")** — that flow stays as in-chat (model writes the translation in the response). The job system is specifically for "I want a downloadable file."

---

## Audit logging (HIPAA)

Every job creation logs:
```
{
  "event": "translate_job_created",
  "userId": "<sub>",
  "jobId": "tj_...",
  "sourceAttachmentId": "att_...",
  "sourceContentType": "application/pdf",
  "sourceSizeBytes": 123456,
  "targetLanguage": "es"
}
```

Every download logs:
```
{
  "event": "translate_download",
  "userId": "<sub>",
  "jobId": "tj_...",
  "format": "docx"
}
```

Every error logs:
```
{
  "event": "translate_failed",
  "userId": "<sub>",
  "jobId": "tj_...",
  "errorType": "BedrockThrottled",
  "stage": "translating"
}
```

Bedrock token counts (`inputTokensApprox`, `outputTokensApprox`) on success.

Per the existing `JsonFormatter` PHI rules: NEVER log `chunk` text, source
text, or translation output. Counts and metadata only.

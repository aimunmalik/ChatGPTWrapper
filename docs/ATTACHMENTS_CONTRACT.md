# Attachments contract (Phase 6c)

**Single source of truth** for the document-upload feature. Backend, frontend, and Terraform work is scoped to this document — any divergence is a bug.

## Goal

Clinicians upload files (PDFs, Excel, DOCX, images, CSV, TXT) to a specific conversation; Praxis extracts text and includes it in the next turn with Claude. Files can be large (>5MB, commonly 10–50MB) and include Excel spreadsheets and scanned documents.

## User flow

1. User is on a conversation (new or existing) in the chat UI.
2. User clicks a paperclip icon in the composer → OS file picker opens.
3. User selects one or more files. Each file shows as a **chip** below the textarea with filename + size + status (`Uploading…` → `Processing…` → `Ready`).
4. User types a message and hits Send.
5. The chat Lambda includes the extracted text from each ready attachment at the top of the user turn, wrapped in `<attachment filename="...">...</attachment>` XML-style tags, then the user's actual message. Claude sees and reasons over it.
6. Attachments persist on the conversation. They appear as permanent chips above the composer whenever that conversation is active, and can be removed with an × button.

## File types + size cap

| Type | MIME | Extraction method |
|---|---|---|
| PDF | `application/pdf` | Textract `DetectDocumentText` (sync, <5MB per call — for larger PDFs we use `StartDocumentTextDetection` async) |
| PNG | `image/png` | Textract `DetectDocumentText` |
| JPG / JPEG | `image/jpeg` | Textract `DetectDocumentText` |
| XLSX | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` | `openpyxl` — sheet-by-sheet text |
| XLS | `application/vnd.ms-excel` | same as XLSX via `xlrd` or convert-then-parse (XLSX path primary; XLS best-effort) |
| DOCX | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | `python-docx` |
| CSV | `text/csv` | stdlib `csv` |
| TXT | `text/plain` | decode as UTF-8 with latin-1 fallback |

**Max file size**: 50 MB (enforced on the presigned POST condition). Anything larger is rejected at upload.

**Max total extracted text per attachment**: 500 KB (~500k chars). If extraction produces more, we truncate and flag `truncated: true` in the attachment record.

## DynamoDB schema

### New table: `anna-chat-{env}-attachments`

| Attribute | Type | Role |
|---|---|---|
| `userId` | S (PK) | Cognito sub |
| `attachmentId` | S (SK) | `att_` + 16 hex chars |
| `conversationId` | S | GSI PK |
| `createdAt` | N | GSI SK (ms since epoch) |
| `filename` | S | Original filename |
| `contentType` | S | MIME type |
| `sizeBytes` | N | — |
| `s3Key` | S | `attachments/{userId}/{attachmentId}/{sanitizedFilename}` |
| `status` | S | `uploading` \| `extracting` \| `ready` \| `error` |
| `statusMessage` | S | Optional error string when `status=error` |
| `extractedText` | S | Full extracted text (up to 500KB). Omitted when status != ready. |
| `extractedPreview` | S | First 300 chars of extractedText, for UI display |
| `truncated` | Bool | Whether extractedText was truncated |
| `ttl` | N | Epoch seconds; set to createdAt + (MESSAGE_TTL_DAYS × 86400) |

**GSI `conversationId-createdAt-index`**: PK `conversationId`, SK `createdAt`, projection ALL — supports listing attachments for a conversation in creation order.

## S3 bucket: `anna-chat-{env}-attachments-{account}`

- CMK encryption (SSE-KMS) — reuses `alias/anna-chat-{env}-s3` key
- Versioning: enabled
- Object Lock: **governance mode, 90-day retention** (HIPAA audit requirement)
- Public access block: all four flags on
- TLS-only bucket policy
- Lifecycle: current versions expire after 180 days (gives ops a buffer beyond TTL); noncurrent versions after 30 days
- CORS: PUT/POST from CloudFront origin + localhost:5173, headers `content-type`, `x-amz-*`
- S3 event notification: on `ObjectCreated:*` under prefix `attachments/` → extraction Lambda

**Object key**: `attachments/{userId}/{attachmentId}/{sanitizedFilename}`

Server-side: we never trust `filename` from the client for paths. Sanitize: replace non-`[a-zA-Z0-9._-]` with `_`, truncate to 128 chars, keep original extension.

## API

All authenticated via existing API Gateway JWT authorizer. New routes added to the existing HTTP API:

### `POST /attachments/presigned-upload`

Request:
```json
{
  "conversationId": "c_abc123",
  "filename": "treatment_plan.pdf",
  "contentType": "application/pdf",
  "sizeBytes": 1234567
}
```

Response:
```json
{
  "attachmentId": "att_9f2e...",
  "uploadUrl": "https://s3.us-east-1.amazonaws.com/anna-chat-dev-attachments-...",
  "uploadFields": { "key": "...", "Content-Type": "...", "bucket": "...", "x-amz-credential": "...", "policy": "...", "x-amz-signature": "...", "x-amz-date": "...", "x-amz-algorithm": "..." },
  "expiresAt": 1713456789000
}
```

Handler:
- Validates file size ≤ 50MB
- Validates contentType ∈ allowed set
- Creates an attachment row with `status=uploading`
- Returns presigned POST with 15-minute expiry, content-length-range condition matching `sizeBytes`, and a bucket-key prefix matching `attachments/{userId}/{attachmentId}/`

### `GET /conversations/{conversationId}/attachments`

Response:
```json
{
  "attachments": [
    {
      "attachmentId": "att_...",
      "filename": "treatment_plan.pdf",
      "contentType": "application/pdf",
      "sizeBytes": 1234567,
      "status": "ready",
      "statusMessage": null,
      "extractedPreview": "Patient: J.D., age 7, diagnosis...",
      "truncated": false,
      "createdAt": 1713456000000
    }
  ]
}
```

Authorization: caller must own the conversation (userId match).

### `DELETE /attachments/{attachmentId}`

Deletes the DDB row and the S3 object. Returns `{ "deleted": true }`.

### Chat integration

`POST /chat` is extended to include extracted attachments in the Bedrock prompt. The handler:
1. Reads body as before (`message`, `conversationId`, `model`)
2. If `conversationId` exists, queries attachments for that conversation where `status=ready`
3. For each, prepends the user turn with:
   ```
   <attachment filename="foo.pdf" contentType="application/pdf">
   {extractedText}
   </attachment>
   ```
4. Joins all attachment blocks before the user's actual message text

The frontend doesn't pass attachment IDs — it just sends the message and lets the server auto-include all ready attachments for that conversation. Simpler wire format; attachments are implicitly "on" for the whole conversation.

## Extraction Lambda

New Lambda `anna-chat-{env}-extract`. Triggered by S3 `ObjectCreated:*` under `attachments/` prefix.

Flow:
1. Get S3 object metadata + read attachment row from DDB by `s3Key` (secondary lookup via GSI or derived key parsing)
2. Set `status=extracting`
3. Stream object bytes to memory (respect 50MB cap)
4. Dispatch by `contentType`:
   - PDF/PNG/JPG → Textract
   - XLSX → openpyxl, iterate sheets, concatenate rows as tab-separated
   - DOCX → python-docx, concatenate paragraphs + table cells
   - CSV → stdlib csv, tab-separated output
   - TXT → decode bytes
5. If extracted text > 500KB: truncate and set `truncated=true`
6. Write `extractedText` + `extractedPreview` + `status=ready` to DDB
7. On error: set `status=error` and `statusMessage`

Lambda config:
- Python 3.12
- VPC-attached like other Lambdas
- IAM: s3:GetObject on the attachments bucket, kms:Decrypt on the s3 CMK, dynamodb RW on attachments table, textract:DetectDocumentText + textract:StartDocumentTextDetection
- Memory: 2048 MB (Textract + openpyxl can be memory-heavy)
- Timeout: 5 min (sync Textract for PDFs can be slow)

## Environment variables (passed to Lambdas)

Additions to `local.lambda_env` in `infra/envs/dev/backend_compute.tf`:

| Name | Value |
|---|---|
| `ATTACHMENTS_TABLE` | DDB table name from the attachments module |
| `ATTACHMENTS_BUCKET` | S3 bucket name from the attachments module |
| `ATTACHMENTS_MAX_SIZE_BYTES` | `52428800` (50MB) |
| `ATTACHMENTS_MAX_TEXT_BYTES` | `512000` (~500KB) |

## Audit

- CloudTrail data events on the attachments bucket → Object Lock S3 bucket (existing)
- CloudWatch logs: attachment events logged with `userId`, `attachmentId`, `conversationId`, `sizeBytes`, `contentType`, `extractorUsed`, `extractedBytes`, `latencyMs` — **never** the extracted text itself

## Out of scope for Phase 6c

- Document library (reusable across conversations) — Phase 6d or later
- Preview rendering (PDF viewer in chat) — no, stays as a chip
- OCR for handwritten text beyond Textract's default — accept limitation
- File type detection by magic bytes — trust the client's contentType after MIME-list validation (with size cap this is acceptable)
- Virus scanning (GuardDuty Malware Protection) — Phase 5 hardening

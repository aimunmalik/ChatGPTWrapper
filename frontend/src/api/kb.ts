import { apiFetch } from "./client";

/** Source classification — matches the 5-value union in docs/KB_CONTRACT.md. */
export type KbSourceType =
  | "research"
  | "training"
  | "protocol"
  | "parent-training"
  | "other";

/** Ingestion pipeline stages. `ready` and `error` are terminal; the rest are
 *  transient and drive status-polling in useKbDocuments. */
export type KbStatus =
  | "uploading"
  | "extracting"
  | "chunking"
  | "embedding"
  | "ready"
  | "error";

/** Shape of a KB document as returned by GET /kb/documents. Does NOT include
 *  chunk content — chunks are not exposed to the frontend in Phase 7 MVP. */
export interface KbDocument {
  kbDocId: string;
  docTitle: string;
  sourceType: KbSourceType;
  /** Free-form grouping label (e.g. "NDBI Research"). Empty string when
   *  the doc was uploaded without a collection. */
  collection: string;
  filename: string;
  sizeBytes: number;
  status: KbStatus;
  statusMessage?: string | null;
  totalChunks?: number;
  uploadedBy: string;
  createdAt: number;
}

export interface KbPresignedUploadRequest {
  filename: string;
  contentType: string;
  sizeBytes: number;
  docTitle: string;
  sourceType: KbSourceType;
  /** Optional collection. Empty / absent means ungrouped. */
  collection?: string;
}

export interface KbPresignedUpload {
  kbDocId: string;
  uploadUrl: string;
  uploadFields: Record<string, string>;
  expiresAt: number;
}

export function getKbPresignedUpload(
  accessToken: string,
  req: KbPresignedUploadRequest,
): Promise<KbPresignedUpload> {
  return apiFetch<KbPresignedUpload>("/kb/presigned-upload", accessToken, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function listKbDocuments(
  accessToken: string,
): Promise<{ documents: KbDocument[] }> {
  return apiFetch<{ documents: KbDocument[] }>("/kb/documents", accessToken);
}

export function deleteKbDocument(
  accessToken: string,
  kbDocId: string,
): Promise<{ deleted: boolean }> {
  return apiFetch<{ deleted: boolean }>(
    `/kb/documents/${encodeURIComponent(kbDocId)}`,
    accessToken,
    { method: "DELETE" },
  );
}

export interface KbDownloadUrl {
  kbDocId: string;
  url: string;
  /** Millisecond epoch when the presigned URL stops working. ~5 min out. */
  expiresAt: number;
  filename: string;
  docTitle: string;
  contentType: string;
}

/** Fetch a short-lived presigned GET URL for the underlying source file.
 *  Not admin-gated server-side — any authenticated user can open a source
 *  Praxis cited in their chat reply. */
export function getKbDownloadUrl(
  accessToken: string,
  kbDocId: string,
): Promise<KbDownloadUrl> {
  return apiFetch<KbDownloadUrl>(
    `/kb/documents/${encodeURIComponent(kbDocId)}/download`,
    accessToken,
  );
}

/**
 * Directly POSTs a file to S3 using the presigned POST form. Does NOT use
 * apiFetch — S3 rejects the Authorization header on presigned uploads.
 *
 * Mirrors `uploadToS3` in `./attachments.ts` — same presigned-POST contract,
 * just a different bucket.
 */
export async function uploadKbDocToS3(
  uploadUrl: string,
  uploadFields: Record<string, string>,
  file: File,
): Promise<void> {
  const form = new FormData();
  for (const [key, value] of Object.entries(uploadFields)) {
    form.append(key, value);
  }
  // Per S3 presigned-POST spec, the file field MUST be appended last.
  form.append("file", file);

  const resp = await fetch(uploadUrl, { method: "POST", body: form });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(
      `S3 upload failed: ${resp.status} ${resp.statusText}${text ? ` — ${text}` : ""}`,
    );
  }
}

import { apiFetch } from "./client";

export type AttachmentStatus = "uploading" | "extracting" | "ready" | "error";

export interface Attachment {
  attachmentId: string;
  filename: string;
  contentType: string;
  sizeBytes: number;
  status: AttachmentStatus;
  statusMessage: string | null;
  extractedPreview?: string;
  truncated?: boolean;
  createdAt: number;
}

export interface PresignedUpload {
  attachmentId: string;
  uploadUrl: string;
  uploadFields: Record<string, string>;
  expiresAt: number;
}

export interface PresignedUploadRequest {
  conversationId: string;
  filename: string;
  contentType: string;
  sizeBytes: number;
}

export function getPresignedUpload(
  accessToken: string,
  req: PresignedUploadRequest,
): Promise<PresignedUpload> {
  return apiFetch<PresignedUpload>("/attachments/presigned-upload", accessToken, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function listAttachments(
  accessToken: string,
  conversationId: string,
): Promise<{ attachments: Attachment[] }> {
  return apiFetch<{ attachments: Attachment[] }>(
    `/conversations/${encodeURIComponent(conversationId)}/attachments`,
    accessToken,
  );
}

export function deleteAttachment(
  accessToken: string,
  attachmentId: string,
): Promise<{ deleted: boolean }> {
  return apiFetch<{ deleted: boolean }>(
    `/attachments/${encodeURIComponent(attachmentId)}`,
    accessToken,
    { method: "DELETE" },
  );
}

/**
 * Directly POSTs a file to S3 using the presigned POST form. Does NOT use
 * apiFetch — S3 rejects the Authorization header on presigned uploads.
 */
export async function uploadToS3(
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

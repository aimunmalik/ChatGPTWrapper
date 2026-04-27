import { apiFetch } from "./client";

/** ISO-639 codes for the V1 supported languages. Backend rejects anything
 *  else; this union keeps the dropdown honest at compile time. */
export type TranslateLanguage =
  | "es"
  | "zh"
  | "vi"
  | "tl"
  | "ko"
  | "ru"
  | "ar"
  | "fr"
  | "pt"
  | "hi"
  | "de"
  | "ja"
  | "en";

/** Worker pipeline stages — `ready` and `error` are terminal and stop the
 *  JobCard's poll loop. The intermediates drive the human-readable status
 *  copy the user sees while waiting. */
export type TranslateJobStatus =
  | "pending"
  | "extracting"
  | "chunking"
  | "translating"
  | "formatting"
  | "ready"
  | "error";

/** UI-facing record for a translation job. Mirrors the GET /translate/jobs/{id}
 *  response in docs/TRANSLATE_CONTRACT.md. The `downloadDocxUrl` /
 *  `downloadPdfUrl` fields are absolute API paths (NOT presigned URLs) — the
 *  frontend mints a fresh presigned URL via getTranslateDownloadUrl() on each
 *  click so the link doesn't expire mid-poll. */
export interface TranslateJob {
  jobId: string;
  /** Optional because POST /translate/jobs returns the create-time shape and
   *  doesn't echo userId. The poll/list responses do. */
  userId?: string;
  status: TranslateJobStatus;
  statusMessage: string | null;
  sourceFilename: string;
  sourceAttachmentId?: string;
  targetLanguage: TranslateLanguage;
  targetLanguageLabel: string;
  createdAt: number;
  updatedAt?: number;
  downloadDocxUrl?: string | null;
  downloadPdfUrl?: string | null;
}

export interface CreateTranslateJobRequest {
  attachmentId: string;
  targetLanguage: TranslateLanguage;
}

/** Slim response from POST /translate/jobs — enough to seed a JobCard
 *  immediately; the polling endpoint fills in the rest. */
export interface CreateTranslateJobResponse {
  jobId: string;
  status: TranslateJobStatus;
  sourceFilename: string;
  targetLanguageLabel: string;
  createdAt: number;
}

export type TranslateDownloadFormat = "docx" | "pdf";

export interface TranslateDownloadUrl {
  url: string;
  /** Millisecond epoch when the presigned URL stops working. ~5 min out. */
  expiresAt: number;
  filename: string;
  contentType: string;
}

/** UI-facing master list of supported languages. Order matters — Spanish first
 *  because that's by far the most common request in ANNA's clinical population.
 *  Mirrors the V1 supported list in docs/TRANSLATE_CONTRACT.md § Languages. */
export const TRANSLATE_LANGUAGES: {
  code: TranslateLanguage;
  label: string;
  rtl?: boolean;
}[] = [
  { code: "es", label: "Spanish" },
  { code: "zh", label: "Mandarin (Simplified Chinese)" },
  { code: "vi", label: "Vietnamese" },
  { code: "tl", label: "Tagalog" },
  { code: "ko", label: "Korean" },
  { code: "ru", label: "Russian" },
  { code: "ar", label: "Arabic", rtl: true },
  { code: "fr", label: "French" },
  { code: "pt", label: "Portuguese (Brazilian)" },
  { code: "hi", label: "Hindi" },
  { code: "de", label: "German" },
  { code: "ja", label: "Japanese" },
  { code: "en", label: "English" },
];

export function createTranslateJob(
  accessToken: string,
  req: CreateTranslateJobRequest,
): Promise<CreateTranslateJobResponse> {
  return apiFetch<CreateTranslateJobResponse>("/translate/jobs", accessToken, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function getTranslateJob(
  accessToken: string,
  jobId: string,
): Promise<TranslateJob> {
  return apiFetch<TranslateJob>(
    `/translate/jobs/${encodeURIComponent(jobId)}`,
    accessToken,
  );
}

export function listTranslateJobs(
  accessToken: string,
): Promise<{ jobs: TranslateJob[] }> {
  return apiFetch<{ jobs: TranslateJob[] }>("/translate/jobs", accessToken);
}

/** Mint a 5-minute presigned GET URL for the requested artifact. Called on
 *  each download-button click — DO NOT cache the URL across clicks; the
 *  expiry is short and the audit log expects one event per download. */
export function getTranslateDownloadUrl(
  accessToken: string,
  jobId: string,
  format: TranslateDownloadFormat,
): Promise<TranslateDownloadUrl> {
  return apiFetch<TranslateDownloadUrl>(
    `/translate/jobs/${encodeURIComponent(jobId)}/download/${format}`,
    accessToken,
  );
}

/** Statuses that mean the worker is still running — drives JobCard polling. */
const TERMINAL_STATUSES: ReadonlySet<TranslateJobStatus> = new Set<TranslateJobStatus>([
  "ready",
  "error",
]);

export function isTerminalStatus(status: TranslateJobStatus): boolean {
  return TERMINAL_STATUSES.has(status);
}

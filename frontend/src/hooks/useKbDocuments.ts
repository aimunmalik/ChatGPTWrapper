import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  deleteKbDocument,
  getKbPresignedUpload,
  listKbDocuments,
  uploadKbDocToS3,
} from "../api/kb";
import type { KbDocument, KbSourceType, KbStatus } from "../api/kb";

interface Options {
  accessToken: string;
}

export interface BatchItem {
  /** Stable client-side id so React can key the list even if the same
   *  filename appears twice in one batch. */
  clientId: string;
  file: File;
  status: "pending" | "uploading" | "done" | "error";
  error?: string;
}

interface UploadManyArgs {
  files: File[];
  collection: string;
  sourceType: KbSourceType;
}

interface UseKbDocuments {
  documents: KbDocument[];
  isLoading: boolean;
  error: string | null;
  /** Live per-file state for the most recent batch. Resets on each new call. */
  batch: BatchItem[];
  /** True while a batch upload is running. */
  batchUploading: boolean;
  uploadMany: (args: UploadManyArgs) => Promise<void>;
  remove: (kbDocId: string) => Promise<void>;
  refresh: () => Promise<void>;
  /** Distinct, non-empty collection names from server state, sorted. */
  collections: string[];
}

const POLL_INTERVAL_MS = 2000;
const MAX_FILE_BYTES = 100 * 1024 * 1024; // 100 MB — matches KB_CONTRACT validation
const DOC_TITLE_MAX = 200;

const ALLOWED_CONTENT_TYPES = new Set<string>([
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "text/plain",
  "text/csv",
]);

/** Statuses that mean the ingestion pipeline is still running — drives poll. */
const SETTLING_STATUSES: ReadonlySet<KbStatus> = new Set<KbStatus>([
  "uploading",
  "extracting",
  "chunking",
  "embedding",
]);

function isSettling(status: KbStatus): boolean {
  return SETTLING_STATUSES.has(status);
}

/** Infer a contentType when the browser didn't populate file.type (common
 *  for .csv / .txt on some platforms). Falls back to the MIME guesses the
 *  backend validates against. */
function resolveContentType(file: File): string {
  if (file.type) return file.type;
  const name = file.name.toLowerCase();
  if (name.endsWith(".pdf")) return "application/pdf";
  if (name.endsWith(".docx"))
    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
  if (name.endsWith(".csv")) return "text/csv";
  if (name.endsWith(".txt")) return "text/plain";
  return "application/octet-stream";
}

/** Derive a default docTitle from a filename: strip extension, cap length. */
function titleFromFilename(filename: string): string {
  return filename.replace(/\.[^.]+$/, "").slice(0, DOC_TITLE_MAX);
}

function newClientId(): string {
  // crypto.randomUUID is fine in modern browsers; fall back to Math.random
  // for older ones so dev & test environments don't crash.
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `cid-${Math.random().toString(36).slice(2)}-${Date.now()}`;
}

export function useKbDocuments({ accessToken }: Options): UseKbDocuments {
  const [documents, setDocuments] = useState<KbDocument[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [batch, setBatch] = useState<BatchItem[]>([]);
  const [batchUploading, setBatchUploading] = useState(false);

  const tokenRef = useRef(accessToken);
  useEffect(() => {
    tokenRef.current = accessToken;
  }, [accessToken]);

  const refresh = useCallback(async () => {
    const token = tokenRef.current;
    if (!token) {
      setDocuments([]);
      return;
    }
    try {
      const resp = await listKbDocuments(token);
      setDocuments(resp.documents);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load KB documents");
    }
  }, []);

  // Initial load whenever the access token changes.
  useEffect(() => {
    let cancelled = false;
    if (!accessToken) {
      setDocuments([]);
      setIsLoading(false);
      return () => {
        cancelled = true;
      };
    }
    setIsLoading(true);
    setError(null);
    listKbDocuments(accessToken)
      .then((resp) => {
        if (!cancelled) setDocuments(resp.documents);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load KB documents");
          setDocuments([]);
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken]);

  // Poll while any doc is still ingesting. Mirrors useAttachmentUpload.
  useEffect(() => {
    const anySettling = documents.some((d) => isSettling(d.status));
    if (!anySettling || !accessToken) return;

    const handle = window.setInterval(() => {
      void refresh();
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(handle);
  }, [documents, accessToken, refresh]);

  const uploadMany = useCallback(
    async ({ files, collection, sourceType }: UploadManyArgs) => {
      const token = tokenRef.current;
      if (!token || files.length === 0) return;

      // Seed the batch state with client-side ids, then mutate per-file as
      // each upload progresses. The UI renders off this array directly.
      const items: BatchItem[] = files.map((f) => ({
        clientId: newClientId(),
        file: f,
        status: "pending",
      }));
      setBatch(items);
      setBatchUploading(true);
      setError(null);

      // Pre-validate everything up front (size, content type) so the admin
      // sees all the rejections immediately instead of trickling in.
      const validated = items.map((item) => {
        if (item.file.size > MAX_FILE_BYTES) {
          return {
            ...item,
            status: "error" as const,
            error: `Too large (${(item.file.size / 1024 / 1024).toFixed(1)} MB, max 100 MB)`,
          };
        }
        const contentType = resolveContentType(item.file);
        if (!ALLOWED_CONTENT_TYPES.has(contentType)) {
          return {
            ...item,
            status: "error" as const,
            error: "Unsupported type — only PDF, DOCX, TXT, CSV",
          };
        }
        return item;
      });
      setBatch(validated);

      // Sequential upload. Parallelism would be nicer but keeps presigned-URL
      // quota, CSP surface, and Textract concurrency predictable. 20 files ×
      // ~2s each = tolerable.
      for (const item of validated) {
        if (item.status === "error") continue;
        setBatch((prev) =>
          prev.map((b) =>
            b.clientId === item.clientId ? { ...b, status: "uploading" } : b,
          ),
        );
        try {
          const contentType = resolveContentType(item.file);
          const docTitle = titleFromFilename(item.file.name);
          const presigned = await getKbPresignedUpload(token, {
            filename: item.file.name,
            contentType,
            sizeBytes: item.file.size,
            docTitle,
            sourceType,
            collection: collection.trim() || undefined,
          });
          await uploadKbDocToS3(
            presigned.uploadUrl,
            presigned.uploadFields,
            item.file,
          );
          setBatch((prev) =>
            prev.map((b) =>
              b.clientId === item.clientId ? { ...b, status: "done" } : b,
            ),
          );
        } catch (err) {
          const msg = err instanceof Error ? err.message : "Upload failed";
          setBatch((prev) =>
            prev.map((b) =>
              b.clientId === item.clientId
                ? { ...b, status: "error", error: msg }
                : b,
            ),
          );
        }
      }

      setBatchUploading(false);
      // Refresh the server list once, so the new META rows show up even if
      // the ingest pipeline hasn't started yet.
      await refresh();
    },
    [refresh],
  );

  const remove = useCallback(
    async (kbDocId: string) => {
      const token = tokenRef.current;
      if (!token) return;
      const snapshot = documents;
      setDocuments((prev) => prev.filter((d) => d.kbDocId !== kbDocId));
      try {
        await deleteKbDocument(token, kbDocId);
      } catch (err) {
        // Rollback and resync with server.
        setDocuments(snapshot);
        setError(err instanceof Error ? err.message : "Failed to delete document");
        await refresh();
      }
    },
    [documents, refresh],
  );

  const collections = useMemo(() => {
    const names = new Set<string>();
    for (const d of documents) {
      if (d.collection) names.add(d.collection);
    }
    return Array.from(names).sort((a, b) => a.localeCompare(b));
  }, [documents]);

  return {
    documents,
    isLoading,
    error,
    batch,
    batchUploading,
    uploadMany,
    remove,
    refresh,
    collections,
  };
}

import { useCallback, useEffect, useRef, useState } from "react";

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

interface UploadArgs {
  file: File;
  docTitle: string;
  sourceType: KbSourceType;
}

interface UseKbDocuments {
  documents: KbDocument[];
  isLoading: boolean;
  error: string | null;
  upload: (args: UploadArgs) => Promise<void>;
  remove: (kbDocId: string) => Promise<void>;
  refresh: () => Promise<void>;
}

const POLL_INTERVAL_MS = 2000;
const MAX_FILE_BYTES = 100 * 1024 * 1024; // 100 MB — matches KB_CONTRACT validation

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

export function useKbDocuments({ accessToken }: Options): UseKbDocuments {
  const [documents, setDocuments] = useState<KbDocument[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  const upload = useCallback(
    async ({ file, docTitle, sourceType }: UploadArgs) => {
      const token = tokenRef.current;
      if (!token) return;

      if (file.size > MAX_FILE_BYTES) {
        window.alert(
          `File is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). The maximum is 100 MB.`,
        );
        return;
      }

      const contentType = resolveContentType(file);
      if (!ALLOWED_CONTENT_TYPES.has(contentType)) {
        window.alert(
          "Unsupported file type. Only PDF, DOCX, TXT, and CSV are accepted.",
        );
        return;
      }

      try {
        const presigned = await getKbPresignedUpload(token, {
          filename: file.name,
          contentType,
          sizeBytes: file.size,
          docTitle,
          sourceType,
        });
        await uploadKbDocToS3(presigned.uploadUrl, presigned.uploadFields, file);
        await refresh();
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Upload failed";
        setError(msg);
        window.alert(`Upload failed: ${msg}`);
      }
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

  return { documents, isLoading, error, upload, remove, refresh };
}

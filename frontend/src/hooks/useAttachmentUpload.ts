import { useCallback, useEffect, useRef, useState } from "react";

import {
  deleteAttachment,
  getPresignedUpload,
  listAttachments,
  uploadToS3,
} from "../api/attachments";
import type { Attachment } from "../api/attachments";

interface Options {
  accessToken: string;
  conversationId: string | null;
}

interface UseAttachmentUpload {
  attachments: Attachment[];
  isLoading: boolean;
  uploadFiles: (files: File[]) => Promise<void>;
  removeAttachment: (attachmentId: string) => Promise<void>;
  refresh: () => Promise<void>;
}

const POLL_INTERVAL_MS = 2000;

function isSettling(status: Attachment["status"]): boolean {
  return status === "uploading" || status === "extracting";
}

export function useAttachmentUpload({
  accessToken,
  conversationId,
}: Options): UseAttachmentUpload {
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Stable access to the latest token/conversationId inside async callbacks.
  const tokenRef = useRef(accessToken);
  const convRef = useRef(conversationId);
  useEffect(() => {
    tokenRef.current = accessToken;
  }, [accessToken]);
  useEffect(() => {
    convRef.current = conversationId;
  }, [conversationId]);

  const refresh = useCallback(async () => {
    const token = tokenRef.current;
    const convId = convRef.current;
    if (!token || !convId) {
      setAttachments([]);
      return;
    }
    try {
      const resp = await listAttachments(token, convId);
      // Only apply if the conversation is still the one we queried for.
      if (convRef.current !== convId) return;
      setAttachments((prev) => {
        // Preserve any optimistic locally-created chips that the server
        // hasn't yet echoed back (e.g. pre-presign race). They have ids
        // that start with "local-".
        const serverIds = new Set(resp.attachments.map((a) => a.attachmentId));
        const locals = prev.filter(
          (a) => a.attachmentId.startsWith("local-") && !serverIds.has(a.attachmentId),
        );
        return [...resp.attachments, ...locals];
      });
    } catch {
      // Swallow; the caller surfaces errors via alerts where needed.
    }
  }, []);

  // Load on mount / when conversationId changes.
  useEffect(() => {
    let cancelled = false;
    if (!accessToken || !conversationId) {
      setAttachments([]);
      setIsLoading(false);
      return () => {
        cancelled = true;
      };
    }
    setIsLoading(true);
    listAttachments(accessToken, conversationId)
      .then((resp) => {
        if (!cancelled) setAttachments(resp.attachments);
      })
      .catch(() => {
        if (!cancelled) setAttachments([]);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken, conversationId]);

  // Poll while any attachment is still settling.
  useEffect(() => {
    const anySettling = attachments.some((a) => isSettling(a.status));
    if (!anySettling || !accessToken || !conversationId) return;

    const handle = window.setInterval(() => {
      void refresh();
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(handle);
  }, [attachments, accessToken, conversationId, refresh]);

  const uploadFiles = useCallback(
    async (files: File[]) => {
      const token = tokenRef.current;
      const convId = convRef.current;
      if (!token || !convId || files.length === 0) return;

      // For each file: presign → optimistic chip → S3 upload → refresh.
      await Promise.all(
        files.map(async (file) => {
          const localId = `local-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
          const optimistic: Attachment = {
            attachmentId: localId,
            filename: file.name,
            contentType: file.type || "application/octet-stream",
            sizeBytes: file.size,
            status: "uploading",
            statusMessage: null,
            createdAt: Date.now(),
          };
          setAttachments((prev) => [...prev, optimistic]);

          try {
            const presigned = await getPresignedUpload(token, {
              conversationId: convId,
              filename: file.name,
              contentType: file.type || "application/octet-stream",
              sizeBytes: file.size,
            });
            await uploadToS3(presigned.uploadUrl, presigned.uploadFields, file);
            // Swap our optimistic id for the real attachmentId so the next
            // refresh() cleanly takes over.
            setAttachments((prev) =>
              prev.map((a) =>
                a.attachmentId === localId
                  ? { ...a, attachmentId: presigned.attachmentId }
                  : a,
              ),
            );
          } catch (err) {
            const msg = err instanceof Error ? err.message : "Upload failed";
            setAttachments((prev) =>
              prev.map((a) =>
                a.attachmentId === localId
                  ? { ...a, status: "error", statusMessage: msg }
                  : a,
              ),
            );
          }
        }),
      );

      // Pick up real server state (extraction status etc.)
      await refresh();
    },
    [refresh],
  );

  const removeAttachment = useCallback(
    async (attachmentId: string) => {
      const token = tokenRef.current;
      if (!token) return;
      const snapshot = attachments;
      setAttachments((prev) => prev.filter((a) => a.attachmentId !== attachmentId));
      try {
        await deleteAttachment(token, attachmentId);
      } catch {
        // Roll back on error.
        setAttachments(snapshot);
        await refresh();
      }
    },
    [attachments, refresh],
  );

  return { attachments, isLoading, uploadFiles, removeAttachment, refresh };
}

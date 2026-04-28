import { useCallback, useEffect, useRef, useState } from "react";

import {
  createTranslateJob,
  listTranslateJobs,
} from "../api/translate";
import type {
  CreateTranslateJobRequest,
  TranslateJob,
} from "../api/translate";

interface Options {
  accessToken: string;
}

interface UseTranslateJobs {
  /** In-session JobCards. Newest jobs first. Cleared on conversation switch
   *  by the parent — the hook itself doesn't care which conversation
   *  produced these jobs. */
  jobs: TranslateJob[];
  /** True while the initial server fetch / refresh is in flight. */
  isLoading: boolean;
  error: string | null;
  /** Submit a new translation job and append its JobCard to local state.
   *  Returns the newly-created job (status="pending") so the caller can
   *  pass it through onJobCreated for any extra wiring. Throws on API
   *  failure — surfacing the error is the caller's responsibility. */
  createJob: (req: CreateTranslateJobRequest, sourceFilenameFallback?: string) => Promise<TranslateJob>;
  /** Remove a JobCard from local state. The contract intentionally treats
   *  this as cosmetic — the worker keeps running on the backend. */
  dismissJob: (jobId: string) => void;
  /** Replace a job in local state with a freshly-polled copy. Called by
   *  individual JobCards as they poll. Used here so the hook's `jobs` array
   *  stays in sync with what each card last saw — drives the
   *  RecentTranslations panel's eventual consistency. */
  updateJob: (job: TranslateJob) => void;
  /** Re-fetch the server list. The list-state is owned by RecentTranslations,
   *  but we expose this here so the modal and the in-chat cards share one
   *  source of truth instead of double-fetching. */
  refresh: () => Promise<void>;
}

/** Local seed for a job we just created via POST. The backend's
 *  CreateTranslateJobResponse is intentionally slim (no userId, no
 *  download URLs yet, no statusMessage) — round-trip to the full shape
 *  by filling the missing fields with sensible nulls. The first poll
 *  cycle replaces this with the canonical server view. */
function seedJob(
  jobId: string,
  status: TranslateJob["status"],
  sourceFilename: string,
  targetLanguage: TranslateJob["targetLanguage"],
  targetLanguageLabel: string,
  createdAt: number,
): TranslateJob {
  return {
    jobId,
    status,
    statusMessage: null,
    sourceFilename,
    targetLanguage,
    targetLanguageLabel,
    createdAt,
    updatedAt: createdAt,
    downloadDocxUrl: null,
    downloadPdfUrl: null,
  };
}

export function useTranslateJobs({ accessToken }: Options): UseTranslateJobs {
  const [jobs, setJobs] = useState<TranslateJob[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const tokenRef = useRef(accessToken);
  useEffect(() => {
    tokenRef.current = accessToken;
  }, [accessToken]);

  const refresh = useCallback(async () => {
    const token = tokenRef.current;
    if (!token) {
      setJobs([]);
      return;
    }
    try {
      const resp = await listTranslateJobs(token);
      setJobs(resp.jobs);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load translations");
    }
  }, []);

  // Per docs/TRANSLATE_CONTRACT.md the JobCard list is IN-SESSION state.
  // Refreshing the page should leave the chat clean — the persistent
  // record of all translations lives in the "Recent translations" modal
  // (which has its own fetcher). We deliberately do NOT auto-fetch the
  // user's job history here on mount, otherwise every old "Ready" card
  // re-appears above the composer on every refresh.
  //
  // `refresh()` is still exposed on the returned API for callers that
  // explicitly want to reconcile (e.g. after creating a job, the hook's
  // own createJob path doesn't need it because the create response is
  // the source of truth).
  useEffect(() => {
    if (!accessToken) {
      setJobs([]);
    }
    setIsLoading(false);
  }, [accessToken]);

  const createJob = useCallback(
    async (
      req: CreateTranslateJobRequest,
      sourceFilenameFallback?: string,
    ): Promise<TranslateJob> => {
      const token = tokenRef.current;
      if (!token) throw new Error("Not authenticated");
      const resp = await createTranslateJob(token, req);
      const job = seedJob(
        resp.jobId,
        resp.status,
        resp.sourceFilename || sourceFilenameFallback || "",
        req.targetLanguage,
        resp.targetLanguageLabel,
        resp.createdAt,
      );
      // Newest first — matches the ordering RecentTranslations uses.
      setJobs((prev) => {
        if (prev.some((j) => j.jobId === job.jobId)) return prev;
        return [job, ...prev];
      });
      return job;
    },
    [],
  );

  const dismissJob = useCallback((jobId: string) => {
    setJobs((prev) => prev.filter((j) => j.jobId !== jobId));
  }, []);

  const updateJob = useCallback((job: TranslateJob) => {
    setJobs((prev) => {
      const idx = prev.findIndex((j) => j.jobId === job.jobId);
      if (idx === -1) return [job, ...prev];
      const next = prev.slice();
      next[idx] = { ...prev[idx], ...job };
      return next;
    });
  }, []);

  return { jobs, isLoading, error, createJob, dismissJob, updateJob, refresh };
}

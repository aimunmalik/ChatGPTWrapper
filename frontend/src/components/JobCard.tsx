import { useEffect, useRef, useState } from "react";
import clsx from "clsx";

import {
  getTranslateDownloadUrl,
  getTranslateJob,
  isTerminalStatus,
} from "../api/translate";
import type {
  TranslateDownloadFormat,
  TranslateJob,
  TranslateJobStatus,
} from "../api/translate";

interface Props {
  job: TranslateJob;
  onJobUpdated: (job: TranslateJob) => void;
  onDismiss: () => void;
  accessToken: string;
}

const POLL_INTERVAL_MS = 5000;

const STATUS_COPY: Record<TranslateJobStatus, string> = {
  pending: "Queued — waiting to start…",
  extracting: "Reading the source document…",
  chunking: "Splitting into translation windows…",
  translating: "Translating…",
  formatting: "Formatting .docx and .pdf outputs…",
  ready: "Translation ready",
  error: "Translation failed",
};

const STATUS_LABEL: Record<TranslateJobStatus, string> = {
  pending: "Queued",
  extracting: "Extracting",
  chunking: "Chunking",
  translating: "Translating",
  formatting: "Formatting",
  ready: "Ready",
  error: "Error",
};

/**
 * Inline translation-status card rendered above the chat thread. Lives in
 * local session state — does NOT persist across page refreshes (the
 * RecentTranslations modal is the durable record). Polls every 5s while
 * the worker is still running; cleans up the poll the moment the status
 * goes terminal so we don't burn API calls forever on a stuck job.
 */
export function JobCard({ job, onJobUpdated, onDismiss, accessToken }: Props) {
  const [downloading, setDownloading] = useState<TranslateDownloadFormat | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  // Stash the latest accessToken without re-running the poll effect — the
  // user might re-auth mid-job and we want subsequent polls to use the
  // refreshed token without restarting the interval.
  const tokenRef = useRef(accessToken);
  useEffect(() => {
    tokenRef.current = accessToken;
  }, [accessToken]);

  // Stable reference to the latest onJobUpdated so the poll effect depends
  // only on the jobId + terminal status, not on identity churn from the
  // parent re-rendering on every state update.
  const updatedRef = useRef(onJobUpdated);
  useEffect(() => {
    updatedRef.current = onJobUpdated;
  }, [onJobUpdated]);

  const status = job.status;
  const terminal = isTerminalStatus(status);

  useEffect(() => {
    if (terminal) return;
    const token = tokenRef.current;
    if (!token) return;

    let cancelled = false;
    const tick = async () => {
      try {
        const fresh = await getTranslateJob(tokenRef.current, job.jobId);
        if (cancelled) return;
        updatedRef.current(fresh);
      } catch {
        // Swallow transient poll errors — the next tick retries. A persistent
        // failure surfaces visually because status never advances.
      }
    };

    const handle = window.setInterval(() => {
      void tick();
    }, POLL_INTERVAL_MS);
    // First tick fires after the interval; for snappier feedback we don't
    // call tick() immediately — the parent already has fresh state from
    // either the create-response seed or a prior poll.
    return () => {
      cancelled = true;
      window.clearInterval(handle);
    };
  }, [job.jobId, terminal]);

  async function handleDownload(format: TranslateDownloadFormat) {
    if (downloading) return;
    setDownloadError(null);
    setDownloading(format);

    // Pre-open a blank tab synchronously — popup blockers only allow
    // window.open during a direct user gesture. We swap the URL in once
    // the presigned link comes back. Mirrors MessageSources.handleOpen.
    const placeholder = window.open("about:blank", "_blank");

    try {
      const resp = await getTranslateDownloadUrl(
        tokenRef.current,
        job.jobId,
        format,
      );
      if (placeholder) {
        placeholder.location.href = resp.url;
      } else {
        const fallback = window.open(resp.url, "_blank");
        if (!fallback) {
          setDownloadError(
            "Your browser blocked the popup. Enable popups for this site and click again.",
          );
        }
      }
    } catch (err) {
      if (placeholder) placeholder.close();
      const msg = err instanceof Error ? err.message : "Download failed";
      setDownloadError(msg);
    } finally {
      setDownloading(null);
    }
  }

  return (
    <div
      className={clsx("job-card", `job-card--${status}`)}
      role="status"
      aria-live="polite"
    >
      <div className="job-card__icon" aria-hidden="true">
        🌐
      </div>
      <div className="job-card__main">
        <div className="job-card__title">
          {status === "ready"
            ? `Translation ready — ${baseFilename(job.sourceFilename)} (${job.targetLanguageLabel})`
            : status === "error"
              ? `Translation failed — ${baseFilename(job.sourceFilename)} (${job.targetLanguageLabel})`
              : `Translating "${baseFilename(job.sourceFilename)}" to ${job.targetLanguageLabel}`}
        </div>
        <div className="job-card__meta">
          <span className={clsx("job-card__status", `job-card__status--${status}`)}>
            {STATUS_LABEL[status]}
          </span>
          <span className="job-card__detail">
            {status === "error" && job.statusMessage
              ? job.statusMessage
              : STATUS_COPY[status]}
          </span>
        </div>

        {status === "ready" && (
          <div className="job-card__downloads">
            {job.downloadDocxUrl && (
              <button
                type="button"
                className="btn btn--primary job-card__download"
                onClick={() => void handleDownload("docx")}
                disabled={downloading !== null}
              >
                {downloading === "docx" ? "Preparing…" : "Download .docx"}
              </button>
            )}
            {job.downloadPdfUrl && (
              <button
                type="button"
                className="btn btn--primary job-card__download"
                onClick={() => void handleDownload("pdf")}
                disabled={downloading !== null}
              >
                {downloading === "pdf" ? "Preparing…" : "Download .pdf"}
              </button>
            )}
            {!job.downloadPdfUrl && job.downloadDocxUrl && (
              // PDF rendering can fail on documents with very wide / complex
              // tables (reportlab LayoutError). The worker uploads the .docx
              // first so the user still gets the Word file in this case.
              <span className="job-card__pdf-note">
                .pdf unavailable for this document — see .docx
              </span>
            )}
          </div>
        )}

        {downloadError && (
          <div className="job-card__error">{downloadError}</div>
        )}
      </div>
      <button
        type="button"
        className="job-card__dismiss"
        onClick={onDismiss}
        aria-label="Dismiss"
        title={
          terminal
            ? "Remove from this conversation"
            : "Hide this card — translation will continue in the background"
        }
      >
        ×
      </button>
    </div>
  );
}

/** Trim very long filenames in the title row — full name still shows in
 *  the source's tooltip on the AttachmentChip if the user needs it. */
function baseFilename(name: string, max = 60): string {
  if (name.length <= max) return name;
  const dot = name.lastIndexOf(".");
  if (dot > 0 && dot > name.length - 8) {
    const ext = name.slice(dot);
    const head = name.slice(0, Math.max(1, max - ext.length - 1));
    return `${head}…${ext}`;
  }
  return `${name.slice(0, max - 1)}…`;
}

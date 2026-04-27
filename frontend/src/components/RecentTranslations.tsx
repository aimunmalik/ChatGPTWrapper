import { useEffect, useState } from "react";
import clsx from "clsx";

import {
  getTranslateDownloadUrl,
  listTranslateJobs,
} from "../api/translate";
import type {
  TranslateDownloadFormat,
  TranslateJob,
  TranslateJobStatus,
} from "../api/translate";

interface Props {
  open: boolean;
  onClose: () => void;
  accessToken: string;
}

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
 * Modal listing the caller's last 50 translation jobs. Mirrors KnowledgeBase
 * visually — same `cmdk-overlay` + `cmdk-panel` shell, same status pill +
 * action layout. Re-fetches on open so freshly-created jobs from this
 * session show up alongside historical ones.
 */
export function RecentTranslations({ open, onClose, accessToken }: Props) {
  const [jobs, setJobs] = useState<TranslateJob[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Esc closes.
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  // Re-fetch on open. Closed → no fetch, so leaving the modal closed is
  // free.
  useEffect(() => {
    if (!open || !accessToken) return;
    let cancelled = false;
    setIsLoading(true);
    setError(null);
    listTranslateJobs(accessToken)
      .then((resp) => {
        if (!cancelled) setJobs(resp.jobs);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load translations");
          setJobs([]);
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, accessToken]);

  if (!open) return null;

  return (
    <div
      className="cmdk-overlay recent-translations"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="cmdk-panel recent-translations__panel"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Recent translations"
        aria-modal="true"
      >
        <div className="recent-translations__header">
          <h2 className="recent-translations__title">Recent translations</h2>
          <button
            type="button"
            className="recent-translations__close"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div className="recent-translations__body">
          {error && <div className="recent-translations__error">{error}</div>}

          {isLoading && jobs.length === 0 ? (
            <div className="recent-translations__empty">Loading…</div>
          ) : jobs.length === 0 ? (
            <div className="recent-translations__empty">
              No translations yet. Use the ⋯ menu on any attached document
              to translate it.
            </div>
          ) : (
            <ul className="recent-translations__list">
              {jobs.map((job) => (
                <RecentJobRow
                  key={job.jobId}
                  job={job}
                  accessToken={accessToken}
                />
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

interface RowProps {
  job: TranslateJob;
  accessToken: string;
}

function RecentJobRow({ job, accessToken }: RowProps) {
  const [downloading, setDownloading] = useState<TranslateDownloadFormat | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  async function handleDownload(format: TranslateDownloadFormat) {
    if (downloading) return;
    setDownloadError(null);
    setDownloading(format);
    const placeholder = window.open("about:blank", "_blank");
    try {
      const resp = await getTranslateDownloadUrl(accessToken, job.jobId, format);
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
    <li className="recent-translations__item">
      <div className="recent-translations__item-main">
        <span
          className="recent-translations__item-title"
          title={job.sourceFilename}
        >
          {job.sourceFilename}
        </span>
        <span className="recent-translations__item-meta">
          <span className="recent-translations__item-language">
            → {job.targetLanguageLabel}
          </span>
          <span
            className={clsx(
              "recent-translations__status",
              `recent-translations__status--${job.status}`,
            )}
            title={
              job.status === "error" && job.statusMessage
                ? job.statusMessage
                : STATUS_LABEL[job.status]
            }
          >
            {STATUS_LABEL[job.status]}
          </span>
          <span className="recent-translations__item-date">
            {formatDate(job.createdAt)}
          </span>
        </span>
        {downloadError && (
          <span className="recent-translations__row-error">{downloadError}</span>
        )}
      </div>
      {job.status === "ready" && (
        <div className="recent-translations__item-actions">
          <button
            type="button"
            className="btn btn--primary recent-translations__download"
            onClick={() => void handleDownload("docx")}
            disabled={downloading !== null}
          >
            {downloading === "docx" ? "Preparing…" : ".docx"}
          </button>
          <button
            type="button"
            className="btn btn--primary recent-translations__download"
            onClick={() => void handleDownload("pdf")}
            disabled={downloading !== null}
          >
            {downloading === "pdf" ? "Preparing…" : ".pdf"}
          </button>
        </div>
      )}
    </li>
  );
}

function formatDate(ms: number): string {
  const d = new Date(ms);
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

import { useEffect, useState } from "react";

import type { Attachment } from "../api/attachments";
import { TRANSLATE_LANGUAGES } from "../api/translate";
import type { TranslateJob, TranslateLanguage } from "../api/translate";

interface Props {
  open: boolean;
  onClose: () => void;
  /** The source attachment to translate. The dialog renders its filename
   *  read-only so the user can't accidentally retarget the wrong doc. */
  attachment: Attachment;
  accessToken: string;
  /** Called once the backend accepts the job. Parent should append the
   *  returned JobCard above the message list. */
  onJobCreated: (job: TranslateJob) => void;
  /** Submit handler — wired to useTranslateJobs.createJob in the parent.
   *  Kept as a prop (not pulled directly via the hook) so the dialog stays
   *  pure and easy to reason about; the parent owns the in-session list. */
  onSubmit: (
    attachmentId: string,
    targetLanguage: TranslateLanguage,
  ) => Promise<TranslateJob>;
  /** Open the RecentTranslations modal from the footer link. */
  onOpenRecent: () => void;
}

const DEFAULT_LANGUAGE: TranslateLanguage = "es";

export function TranslateDialog({
  open,
  onClose,
  attachment,
  onJobCreated,
  onSubmit,
  onOpenRecent,
}: Props) {
  const [targetLanguage, setTargetLanguage] =
    useState<TranslateLanguage>(DEFAULT_LANGUAGE);
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  // Reset on close so reopening for a different attachment never shows
  // stale form state.
  useEffect(() => {
    if (!open) {
      setTargetLanguage(DEFAULT_LANGUAGE);
      setSubmitting(false);
      setFormError(null);
    }
  }, [open]);

  // Esc closes — but not while the POST is in flight, so the in-session
  // job list doesn't get orphaned half-created.
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !submitting) {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose, submitting]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (submitting) return;

    setFormError(null);
    if (attachment.status !== "ready") {
      setFormError(
        "This attachment isn't fully processed yet. Wait until extraction finishes, then try again.",
      );
      return;
    }

    setSubmitting(true);
    try {
      const job = await onSubmit(attachment.attachmentId, targetLanguage);
      onJobCreated(job);
      onClose();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to start translation";
      setFormError(msg);
    } finally {
      setSubmitting(false);
    }
  }

  if (!open) return null;

  return (
    <div
      className="cmdk-overlay translate-dialog"
      onClick={() => {
        if (!submitting) onClose();
      }}
      role="presentation"
    >
      <div
        className="cmdk-panel translate-dialog__panel"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Translate document"
        aria-modal="true"
      >
        <div className="translate-dialog__header">
          <h2 className="translate-dialog__title">Translate document</h2>
          <button
            type="button"
            className="translate-dialog__close"
            onClick={onClose}
            disabled={submitting}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <form className="translate-dialog__body" onSubmit={handleSubmit}>
          <label className="translate-dialog__field">
            <span className="translate-dialog__field-label">Source</span>
            <input
              type="text"
              className="translate-dialog__input translate-dialog__input--readonly"
              value={attachment.filename}
              readOnly
              tabIndex={-1}
            />
          </label>

          <label className="translate-dialog__field">
            <span className="translate-dialog__field-label">Target language</span>
            <select
              className="translate-dialog__input"
              value={targetLanguage}
              onChange={(e) => setTargetLanguage(e.target.value as TranslateLanguage)}
              disabled={submitting}
            >
              {TRANSLATE_LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label}
                </option>
              ))}
            </select>
            <span className="translate-dialog__field-hint">
              Translation runs in the background — typically 30 seconds to a few
              minutes depending on document length.
            </span>
          </label>

          {formError && (
            <div className="translate-dialog__error">{formError}</div>
          )}

          <div className="translate-dialog__actions">
            <button
              type="button"
              className="btn btn--ghost"
              onClick={onClose}
              disabled={submitting}
            >
              Cancel
            </button>
            <span className="translate-dialog__actions-spacer" />
            <button
              type="submit"
              className="btn btn--primary"
              disabled={submitting}
            >
              {submitting ? "Starting…" : "Start translation"}
            </button>
          </div>

          <div className="translate-dialog__footer">
            <button
              type="button"
              className="translate-dialog__footer-link"
              onClick={() => {
                onClose();
                onOpenRecent();
              }}
              disabled={submitting}
            >
              View past translations →
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

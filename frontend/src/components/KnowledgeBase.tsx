import { useEffect, useRef, useState } from "react";
import clsx from "clsx";

import type { KbDocument, KbSourceType, KbStatus } from "../api/kb";
import { useKbDocuments } from "../hooks/useKbDocuments";

interface Props {
  open: boolean;
  onClose: () => void;
  accessToken: string;
}

const TITLE_MAX = 200;

const SOURCE_TYPE_OPTIONS: { value: KbSourceType; label: string }[] = [
  { value: "research", label: "Research" },
  { value: "training", label: "Training" },
  { value: "protocol", label: "Protocol" },
  { value: "parent-training", label: "Parent training" },
  { value: "other", label: "Other" },
];

const ACCEPT = ".pdf,.docx,.txt,.csv";

export function KnowledgeBase({ open, onClose, accessToken }: Props) {
  const { documents, isLoading, error, upload, remove } = useKbDocuments({
    accessToken,
  });

  const [file, setFile] = useState<File | null>(null);
  const [docTitle, setDocTitle] = useState("");
  const [sourceType, setSourceType] = useState<KbSourceType>("training");
  const [formError, setFormError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Reset the upload form whenever the modal closes so the next open is
  // a clean slate. Keeps the already-uploaded doc list in place (it's a
  // hook, not local state).
  useEffect(() => {
    if (!open) {
      setFile(null);
      setDocTitle("");
      setSourceType("training");
      setFormError(null);
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }, [open]);

  // Esc closes the modal — but only when we're not mid-upload, so the
  // upload state machine doesn't get orphaned.
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !uploading) {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose, uploading]);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = e.target.files?.[0] ?? null;
    setFile(picked);
    // Default docTitle to filename (sans extension) if the admin hasn't
    // already typed something. Easy nudge without overwriting their edits.
    if (picked && !docTitle.trim()) {
      const base = picked.name.replace(/\.[^.]+$/, "");
      setDocTitle(base.slice(0, TITLE_MAX));
    }
  }

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    setFormError(null);

    if (!file) {
      setFormError("Pick a file first.");
      return;
    }
    const title = docTitle.trim();
    if (!title) {
      setFormError("Title is required.");
      return;
    }
    if (title.length > TITLE_MAX) {
      setFormError(`Title must be ${TITLE_MAX} characters or fewer.`);
      return;
    }

    setUploading(true);
    try {
      await upload({ file, docTitle: title, sourceType });
      // Reset form on success; keep the modal open so admins can upload
      // multiple docs in a row.
      setFile(null);
      setDocTitle("");
      setSourceType("training");
      if (fileInputRef.current) fileInputRef.current.value = "";
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(doc: KbDocument) {
    if (!window.confirm(`Delete "${doc.docTitle}"? This can't be undone.`)) {
      return;
    }
    await remove(doc.kbDocId);
  }

  if (!open) return null;

  return (
    <div
      className="cmdk-overlay kb-library"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="cmdk-panel kb-library__panel"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Knowledge base"
        aria-modal="true"
      >
        <div className="kb-library__header">
          <h2 className="kb-library__title">Knowledge base</h2>
          <button
            type="button"
            className="kb-library__close"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div className="kb-library__body">
          <form className="kb-library__form" onSubmit={handleUpload}>
            <label className="kb-library__field">
              <span className="kb-library__field-label">File</span>
              <input
                ref={fileInputRef}
                type="file"
                accept={ACCEPT}
                onChange={handleFileChange}
                disabled={uploading}
                className="kb-library__file-input"
              />
              <span className="kb-library__field-hint">
                PDF, DOCX, TXT, or CSV — up to 100 MB
              </span>
            </label>

            <label className="kb-library__field">
              <span className="kb-library__field-label">Title</span>
              <input
                type="text"
                className="kb-library__input"
                maxLength={TITLE_MAX}
                value={docTitle}
                onChange={(e) => setDocTitle(e.target.value)}
                placeholder="e.g. ANNA Training Module 3"
                disabled={uploading}
              />
              <span className="kb-library__field-hint">
                {docTitle.length}/{TITLE_MAX}
              </span>
            </label>

            <label className="kb-library__field">
              <span className="kb-library__field-label">Source type</span>
              <select
                className="kb-library__input"
                value={sourceType}
                onChange={(e) => setSourceType(e.target.value as KbSourceType)}
                disabled={uploading}
              >
                {SOURCE_TYPE_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>

            {formError && <div className="kb-library__error">{formError}</div>}
            {error && !formError && (
              <div className="kb-library__error">{error}</div>
            )}

            <div className="kb-library__actions">
              <span className="kb-library__actions-spacer" />
              <button
                type="submit"
                className="btn btn--primary"
                disabled={uploading || !file}
              >
                {uploading ? "Uploading…" : "Upload"}
              </button>
            </div>
          </form>

          <div className="kb-library__list-header">
            <h3 className="kb-library__list-title">Uploaded documents</h3>
            <span className="kb-library__list-count">{documents.length}</span>
          </div>

          {isLoading && documents.length === 0 ? (
            <div className="kb-library__empty">Loading…</div>
          ) : documents.length === 0 ? (
            <div className="kb-library__empty">
              No knowledge-base documents yet. Upload a PDF, DOCX, TXT, or
              CSV to make it available to Praxis.
            </div>
          ) : (
            <ul className="kb-library__list">
              {documents.map((d) => (
                <KbDocRow key={d.kbDocId} doc={d} onDelete={handleDelete} />
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

interface RowProps {
  doc: KbDocument;
  onDelete: (doc: KbDocument) => void;
}

function KbDocRow({ doc, onDelete }: RowProps) {
  const canDelete = doc.status === "ready" || doc.status === "error";
  return (
    <li className="kb-library__item">
      <div className="kb-library__item-main">
        <span className="kb-library__item-title" title={doc.docTitle}>
          {doc.docTitle}
        </span>
        <span className="kb-library__item-meta">
          <SourceTypeBadge sourceType={doc.sourceType} />
          <StatusPill status={doc.status} statusMessage={doc.statusMessage} />
          <span className="kb-library__item-date">
            {formatDate(doc.createdAt)}
          </span>
        </span>
      </div>
      <button
        type="button"
        className="kb-library__item-delete"
        onClick={() => onDelete(doc)}
        aria-label={`Delete ${doc.docTitle}`}
        disabled={!canDelete}
        title={canDelete ? "Delete" : "Deletion available once ingestion finishes"}
      >
        ×
      </button>
    </li>
  );
}

function SourceTypeBadge({ sourceType }: { sourceType: KbSourceType }) {
  const label =
    SOURCE_TYPE_OPTIONS.find((o) => o.value === sourceType)?.label ?? sourceType;
  return (
    <span className={clsx("kb-badge", `kb-badge--${sourceType}`)}>{label}</span>
  );
}

function StatusPill({
  status,
  statusMessage,
}: {
  status: KbStatus;
  statusMessage?: string | null;
}) {
  const label = STATUS_LABELS[status];
  const title =
    status === "error" && statusMessage
      ? statusMessage
      : STATUS_LABELS[status];
  return (
    <span
      className={clsx("kb-status", `kb-status--${status}`)}
      title={title}
    >
      {label}
    </span>
  );
}

const STATUS_LABELS: Record<KbStatus, string> = {
  uploading: "Uploading",
  extracting: "Extracting",
  chunking: "Chunking",
  embedding: "Embedding",
  ready: "Ready",
  error: "Error",
};

function formatDate(ms: number): string {
  const d = new Date(ms);
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

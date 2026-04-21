import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";

import type { KbDocument, KbSourceType, KbStatus } from "../api/kb";
import type { BatchItem } from "../hooks/useKbDocuments";
import { useKbDocuments } from "../hooks/useKbDocuments";

interface Props {
  open: boolean;
  onClose: () => void;
  accessToken: string;
}

const COLLECTION_MAX = 80;

const SOURCE_TYPE_OPTIONS: { value: KbSourceType; label: string }[] = [
  { value: "research", label: "Research" },
  { value: "training", label: "Training" },
  { value: "protocol", label: "Protocol" },
  { value: "parent-training", label: "Parent training" },
  { value: "other", label: "Other" },
];

const ACCEPT = ".pdf,.docx,.txt,.csv";
const UNGROUPED_LABEL = "Ungrouped";

export function KnowledgeBase({ open, onClose, accessToken }: Props) {
  const {
    documents,
    isLoading,
    error,
    batch,
    batchUploading,
    uploadMany,
    remove,
    collections,
  } = useKbDocuments({ accessToken });

  const [files, setFiles] = useState<File[]>([]);
  const [collection, setCollection] = useState("");
  const [sourceType, setSourceType] = useState<KbSourceType>("research");
  const [formError, setFormError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Reset the upload form whenever the modal closes so the next open is a
  // clean slate. Keeps the already-uploaded doc list in place (hook state).
  useEffect(() => {
    if (!open) {
      setFiles([]);
      setCollection("");
      setSourceType("research");
      setFormError(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }, [open]);

  // Esc closes — but only when not mid-upload, so the batch state machine
  // doesn't get orphaned.
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !batchUploading) {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose, batchUploading]);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = Array.from(e.target.files ?? []);
    setFiles(picked);
  }

  function removeStagedFile(idx: number) {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  }

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    setFormError(null);

    if (files.length === 0) {
      setFormError("Pick one or more files first.");
      return;
    }
    const col = collection.trim();
    if (col.length > COLLECTION_MAX) {
      setFormError(`Collection name must be ${COLLECTION_MAX} characters or fewer.`);
      return;
    }

    await uploadMany({ files, collection: col, sourceType });

    // Clear the staged files once the batch finishes — the live `batch`
    // array from the hook keeps the visible per-file results. Don't clear
    // the collection; admins often upload several waves into the same one.
    setFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function handleDelete(doc: KbDocument) {
    if (!window.confirm(`Delete "${doc.docTitle}"? This can't be undone.`)) {
      return;
    }
    await remove(doc.kbDocId);
  }

  // Group the uploaded docs for display — by collection, newest first within
  // each group, with "Ungrouped" pinned last.
  const grouped = useMemo(() => {
    const byKey = new Map<string, KbDocument[]>();
    for (const doc of documents) {
      const key = doc.collection || "";
      const list = byKey.get(key) ?? [];
      list.push(doc);
      byKey.set(key, list);
    }
    const keys = Array.from(byKey.keys()).sort((a, b) => {
      if (a === "" && b !== "") return 1;
      if (b === "" && a !== "") return -1;
      return a.localeCompare(b);
    });
    return keys.map((k) => ({
      name: k || UNGROUPED_LABEL,
      docs: (byKey.get(k) ?? []).sort((a, b) => b.createdAt - a.createdAt),
    }));
  }, [documents]);

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
              <span className="kb-library__field-label">Collection</span>
              <input
                type="text"
                className="kb-library__input"
                list="kb-existing-collections"
                maxLength={COLLECTION_MAX}
                value={collection}
                onChange={(e) => setCollection(e.target.value)}
                placeholder="e.g. NDBI Research"
                disabled={batchUploading}
              />
              <datalist id="kb-existing-collections">
                {collections.map((c) => (
                  <option key={c} value={c} />
                ))}
              </datalist>
              <span className="kb-library__field-hint">
                All files in this batch go into this collection. Leave blank
                for ungrouped.
              </span>
            </label>

            <label className="kb-library__field">
              <span className="kb-library__field-label">Source type</span>
              <select
                className="kb-library__input"
                value={sourceType}
                onChange={(e) => setSourceType(e.target.value as KbSourceType)}
                disabled={batchUploading}
              >
                {SOURCE_TYPE_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
              <span className="kb-library__field-hint">
                Applied to every file in this batch.
              </span>
            </label>

            <label className="kb-library__field">
              <span className="kb-library__field-label">Files</span>
              <input
                ref={fileInputRef}
                type="file"
                accept={ACCEPT}
                multiple
                onChange={handleFileChange}
                disabled={batchUploading}
                className="kb-library__file-input"
              />
              <span className="kb-library__field-hint">
                PDF, DOCX, TXT, or CSV — up to 100 MB each. Hold ⌘/Ctrl or
                Shift to pick multiple.
              </span>
            </label>

            {files.length > 0 && (
              <ul className="kb-library__staged">
                {files.map((f, i) => (
                  <li key={`${f.name}-${i}`} className="kb-library__staged-item">
                    <span className="kb-library__staged-name" title={f.name}>
                      {f.name}
                    </span>
                    <span className="kb-library__staged-size">
                      {(f.size / 1024 / 1024).toFixed(1)} MB
                    </span>
                    {!batchUploading && (
                      <button
                        type="button"
                        className="kb-library__staged-remove"
                        onClick={() => removeStagedFile(i)}
                        aria-label={`Remove ${f.name}`}
                      >
                        ×
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            )}

            {batch.length > 0 && (
              <div className="kb-library__batch">
                <div className="kb-library__batch-title">
                  {batchUploading
                    ? `Uploading ${batch.filter((b) => b.status === "done").length}/${batch.length}…`
                    : `Batch complete — ${batch.filter((b) => b.status === "done").length} done, ${batch.filter((b) => b.status === "error").length} failed`}
                </div>
                <ul className="kb-library__batch-list">
                  {batch.map((b) => (
                    <BatchRow key={b.clientId} item={b} />
                  ))}
                </ul>
              </div>
            )}

            {formError && <div className="kb-library__error">{formError}</div>}
            {error && !formError && (
              <div className="kb-library__error">{error}</div>
            )}

            <div className="kb-library__actions">
              <span className="kb-library__actions-spacer" />
              <button
                type="submit"
                className="btn btn--primary"
                disabled={batchUploading || files.length === 0}
              >
                {batchUploading
                  ? "Uploading…"
                  : files.length > 1
                    ? `Upload ${files.length} files`
                    : "Upload"}
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
            <div className="kb-library__groups">
              {grouped.map((g) => (
                <section key={g.name} className="kb-library__group">
                  <header className="kb-library__group-header">
                    <span className="kb-library__group-name">{g.name}</span>
                    <span className="kb-library__group-count">
                      {g.docs.length}
                    </span>
                  </header>
                  <ul className="kb-library__list">
                    {g.docs.map((d) => (
                      <KbDocRow
                        key={d.kbDocId}
                        doc={d}
                        onDelete={handleDelete}
                      />
                    ))}
                  </ul>
                </section>
              ))}
            </div>
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

function BatchRow({ item }: { item: BatchItem }) {
  return (
    <li className={clsx("kb-library__batch-item", `kb-library__batch-item--${item.status}`)}>
      <span className="kb-library__batch-name" title={item.file.name}>
        {item.file.name}
      </span>
      <span className="kb-library__batch-status">
        {item.status === "pending" && "Queued"}
        {item.status === "uploading" && "Uploading…"}
        {item.status === "done" && "Uploaded"}
        {item.status === "error" && (item.error ?? "Failed")}
      </span>
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

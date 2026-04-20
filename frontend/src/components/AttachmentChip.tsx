import clsx from "clsx";

import type { Attachment } from "../api/attachments";

interface Props {
  attachment: Attachment;
  onRemove: (id: string) => void;
  compact?: boolean;
}

function truncate(name: string, max = 28): string {
  if (name.length <= max) return name;
  const dot = name.lastIndexOf(".");
  if (dot > 0 && dot > name.length - 8) {
    const ext = name.slice(dot);
    const head = name.slice(0, Math.max(1, max - ext.length - 1));
    return `${head}…${ext}`;
  }
  return `${name.slice(0, max - 1)}…`;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

export function AttachmentChip({ attachment, onRemove, compact }: Props) {
  const { status, statusMessage, filename, sizeBytes, attachmentId } = attachment;
  const canRemove = status === "ready" || status === "error";

  return (
    <div
      className={clsx("attachment-chip", {
        "attachment-chip--compact": compact,
        [`attachment-chip--${status}`]: true,
      })}
      title={status === "error" && statusMessage ? statusMessage : filename}
    >
      <span className="attachment-chip__name">{truncate(filename)}</span>
      <span className="attachment-chip__size">{formatSize(sizeBytes)}</span>
      <span className={clsx("attachment-chip__badge", `attachment-chip__badge--${status}`)}>
        {status === "uploading" && (
          <>
            <span className="attachment-chip__dot" /> Uploading…
          </>
        )}
        {status === "extracting" && (
          <>
            <span className="attachment-chip__dot attachment-chip__dot--pulse" /> Processing…
          </>
        )}
        {status === "ready" && "Ready"}
        {status === "error" && "Failed"}
      </span>
      {canRemove && (
        <button
          type="button"
          className="attachment-chip__remove"
          onClick={() => onRemove(attachmentId)}
          aria-label={`Remove ${filename}`}
        >
          ×
        </button>
      )}
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import clsx from "clsx";

import type { Attachment } from "../api/attachments";

interface Props {
  attachment: Attachment;
  onRemove: (id: string) => void;
  /** Optional — when present, an overflow `⋯` button appears with a
   *  "Translate…" entry that calls this on click. Hidden entirely when
   *  unset so the chip stays minimal in any context that doesn't wire
   *  translation up (e.g. future read-only views). */
  onTranslate?: () => void;
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

export function AttachmentChip({
  attachment,
  onRemove,
  onTranslate,
  compact,
}: Props) {
  const { status, statusMessage, filename, sizeBytes, attachmentId } = attachment;
  const canRemove = status === "ready" || status === "error";
  // Translation only makes sense once extraction has produced text. We
  // still render the menu item in other states with a disabled hint so
  // users discover the affordance, but actually firing the dialog is
  // gated to ready-only.
  const canTranslate = Boolean(onTranslate) && status === "ready";

  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  // Close the popover on outside click / Esc. Wired only while it's open
  // so we don't leak global listeners on every chip in the tray.
  useEffect(() => {
    if (!menuOpen) return;
    const handleClick = (e: MouseEvent) => {
      const node = menuRef.current;
      if (node && !node.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        setMenuOpen(false);
      }
    };
    window.addEventListener("mousedown", handleClick);
    window.addEventListener("keydown", handleKey);
    return () => {
      window.removeEventListener("mousedown", handleClick);
      window.removeEventListener("keydown", handleKey);
    };
  }, [menuOpen]);

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
      {onTranslate && (
        <div className="attachment-chip__menu" ref={menuRef}>
          <button
            type="button"
            className="attachment-chip__menu-btn"
            onClick={() => setMenuOpen((o) => !o)}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            aria-label={`More actions for ${filename}`}
          >
            ⋯
          </button>
          {menuOpen && (
            <div
              className="attachment-chip__menu-popover"
              role="menu"
            >
              <button
                type="button"
                className="attachment-chip__menu-item"
                role="menuitem"
                disabled={!canTranslate}
                title={
                  canTranslate
                    ? "Translate this document"
                    : "Available once the document finishes processing"
                }
                onClick={() => {
                  setMenuOpen(false);
                  onTranslate?.();
                }}
              >
                Translate…
              </button>
            </div>
          )}
        </div>
      )}
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

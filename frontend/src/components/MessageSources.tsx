import { useState } from "react";
import { useAuth } from "react-oidc-context";

import type { Source } from "../api/chat";
import { getKbDownloadUrl } from "../api/kb";

interface Props {
  sources: Source[];
}

/**
 * Renders a row of numbered source pills below an assistant message when
 * retrieval returned chunks. Each pill is a button: clicking it fetches a
 * short-lived presigned GET URL for the source PDF and opens it in a new
 * tab, jumping to the cited page when we know it.
 *
 * Returns null (not an empty div) when there are no sources, so the
 * surrounding layout doesn't get an unnecessary margin row.
 */
export function MessageSources({ sources }: Props) {
  const auth = useAuth();
  const accessToken = auth.user?.access_token ?? "";
  // Track per-pill loading state so multiple concurrent clicks don't
  // collide visually. Keyed by `index` which is stable per message.
  const [loading, setLoading] = useState<Record<number, boolean>>({});
  const [error, setError] = useState<string | null>(null);

  if (sources.length === 0) return null;

  async function handleOpen(source: Source) {
    if (!accessToken) return;
    setError(null);
    setLoading((prev) => ({ ...prev, [source.index]: true }));

    // Pre-open a blank tab synchronously. Popup blockers only allow
    // window.open during a direct user gesture — doing it after the
    // await on the fetch makes Chrome block it. We navigate the tab
    // to the real URL once we have it, or close it if the API fails.
    const placeholder = window.open("about:blank", "_blank");

    try {
      const resp = await getKbDownloadUrl(accessToken, source.kbDocId);
      // Most PDF viewers (Chrome's native, Adobe Reader, PDF.js) honor
      // #page=N to jump straight to a page. Non-PDF types (docx) ignore
      // it harmlessly — the browser will download instead.
      const href =
        typeof source.pageNumber === "number" && source.pageNumber > 0
          ? `${resp.url}#page=${source.pageNumber}`
          : resp.url;
      if (placeholder) {
        placeholder.location.href = href;
      } else {
        // Popup blocker killed the placeholder open — try the post-fetch
        // window.open as a fallback. Some strict blockers may still
        // reject it, in which case the user sees the error below.
        const fallback = window.open(href, "_blank");
        if (!fallback) {
          setError(
            "Your browser blocked the popup. Enable popups for this site and click again.",
          );
        }
      }
    } catch (err) {
      if (placeholder) placeholder.close();
      const msg =
        err instanceof Error ? err.message : "Couldn't fetch the source link";
      setError(msg);
    } finally {
      setLoading((prev) => ({ ...prev, [source.index]: false }));
    }
  }

  return (
    <div className="message__sources">
      {sources.map((s) => {
        const isLoading = !!loading[s.index];
        // Assistant messages persisted before the download feature shipped
        // don't carry a kbDocId. Render those as a static chip with a
        // tooltip that explains why — more honest than a click that
        // silently does nothing.
        const canOpen = Boolean(s.kbDocId);
        if (!canOpen) {
          return (
            <span
              key={s.index}
              className="message__source-pill message__source-pill--static"
              title="This source was cited before per-message download links were supported. Ask a new question to get clickable sources."
              aria-label={`Source ${s.index}: ${s.docTitle} (not openable — legacy message)`}
            >
              [{s.index}] {s.docTitle}
            </span>
          );
        }
        return (
          <button
            key={s.index}
            type="button"
            className="message__source-pill"
            title={buildTooltip(s)}
            onClick={() => void handleOpen(s)}
            disabled={isLoading || !accessToken}
            aria-label={`Open source ${s.index}: ${s.docTitle}`}
          >
            {isLoading ? "…" : `[${s.index}]`} {s.docTitle}
          </button>
        );
      })}
      {error && <span className="message__sources-error">{error}</span>}
    </div>
  );
}

function buildTooltip(s: Source): string {
  const parts = [s.sourceType];
  if (typeof s.pageNumber === "number") parts.push(`page ${s.pageNumber}`);
  parts.push(`relevance ${scoreBucket(s.score)}`);
  parts.push("click to open");
  return parts.join(" · ");
}

/** Translate a raw cosine score into a coarse bucket — we intentionally
 *  don't show the exact number in the UI (users don't need that precision,
 *  and it invites spurious comparisons). */
function scoreBucket(score: number): string {
  if (score >= 0.75) return "high";
  if (score >= 0.55) return "medium";
  return "low";
}

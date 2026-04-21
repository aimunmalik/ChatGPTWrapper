import type { Source } from "../api/chat";

interface Props {
  sources: Source[];
}

/**
 * Renders a row of numbered source pills below an assistant message when
 * retrieval returned chunks. Links `[1]`, `[2]` etc. in the assistant's
 * body to the underlying doc via hover tooltip — no external nav, no
 * chunk preview (out of scope per KB_CONTRACT MVP).
 *
 * Returns null (not an empty div) when there are no sources, so the
 * surrounding layout doesn't get an unnecessary margin row.
 */
export function MessageSources({ sources }: Props) {
  if (sources.length === 0) return null;
  return (
    <div className="message__sources">
      {sources.map((s) => (
        <span
          key={s.index}
          className="message__source-pill"
          title={buildTooltip(s)}
        >
          [{s.index}] {s.docTitle}
        </span>
      ))}
    </div>
  );
}

function buildTooltip(s: Source): string {
  const parts = [s.sourceType];
  if (typeof s.pageNumber === "number") parts.push(`page ${s.pageNumber}`);
  parts.push(`relevance ${scoreBucket(s.score)}`);
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

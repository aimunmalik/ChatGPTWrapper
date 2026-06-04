import { useEffect, useMemo, useRef, useState } from "react";

import { postChatStream, pollChatStream } from "../api/chat";
import type { Source } from "../api/chat";
import type { Attachment } from "../api/attachments";
import type { MessageSummary } from "../api/conversations";
import type { TranslateLanguage } from "../api/translate";
import { useAttachmentUpload } from "../hooks/useAttachmentUpload";
import { useTranslateJobs } from "../hooks/useTranslateJobs";
import { AttachmentChip } from "./AttachmentChip";
import { AttachmentPicker } from "./AttachmentPicker";
import { JobCard } from "./JobCard";
import { DEFAULT_MODEL, ModelPicker } from "./ModelPicker";
import { Message } from "./Message";
import { MessagesSkeleton } from "./Skeleton";
import { TranslateDialog } from "./TranslateDialog";

const MODEL_STORAGE_KEY = "anna-chat:model";

interface Props {
  conversationId: string | null;
  initialMessages: MessageSummary[];
  loading: boolean;
  accessToken: string;
  onConversationCreated: (conversationId: string, title: string) => void;
}

interface DraftMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  pending?: boolean;
  sources?: Source[];
}

export function ChatView({
  conversationId,
  initialMessages,
  loading,
  accessToken,
  onConversationCreated,
}: Props) {
  const [drafts, setDrafts] = useState<DraftMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState<string>(
    () => window.localStorage.getItem(MODEL_STORAGE_KEY) ?? DEFAULT_MODEL,
  );
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const activeConvRef = useRef<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Holds the in-flight streaming poll so it can be cancelled on unmount or
  // when the conversation switches (mirrors JobCard's setInterval + cancelled
  // flag style). `cancelled` short-circuits any tick already scheduled;
  // `settle` resolves the awaiting promise in handleSubmit so it doesn't dangle.
  const pollRef = useRef<{
    handle: number;
    cancelled: boolean;
    settle: () => void;
  } | null>(null);
  const stopPoll = () => {
    if (pollRef.current) {
      const ctrl = pollRef.current;
      ctrl.cancelled = true;
      window.clearInterval(ctrl.handle);
      pollRef.current = null;
      ctrl.settle();
    }
  };

  const {
    attachments,
    uploadFiles,
    removeAttachment,
  } = useAttachmentUpload({
    accessToken,
    conversationId,
    onConversationCreated,
  });

  const {
    jobs: translateJobs,
    createJob: createTranslateJob,
    dismissJob: dismissTranslateJob,
    updateJob: updateTranslateJob,
  } = useTranslateJobs({ accessToken });

  // Translation modal state. translateTarget is the chip the user opened
  // the dialog from; null means the dialog is closed.
  const [translateTarget, setTranslateTarget] = useState<Attachment | null>(null);

  useEffect(() => {
    window.localStorage.setItem(MODEL_STORAGE_KEY, model);
  }, [model]);

  // Listen for model changes triggered by the command palette.
  useEffect(() => {
    const handler = () => {
      const stored = window.localStorage.getItem(MODEL_STORAGE_KEY);
      if (stored) setModel(stored);
    };
    window.addEventListener("praxis:model-changed", handler);
    return () => window.removeEventListener("praxis:model-changed", handler);
  }, []);

  // Listen for prompt template insertions from the command palette.
  useEffect(() => {
    const handler = (evt: Event) => {
      const detail = (evt as CustomEvent<{ text?: string }>).detail;
      const text = detail?.text;
      if (!text) return;
      setInput(text);
      queueMicrotask(() => {
        const el = textareaRef.current;
        if (!el) return;
        el.focus();
        const firstBracket = text.indexOf("[");
        if (firstBracket !== -1) {
          el.setSelectionRange(firstBracket, firstBracket);
        } else {
          el.setSelectionRange(el.value.length, el.value.length);
        }
      });
    };
    window.addEventListener("praxis:insert-prompt", handler);
    return () => window.removeEventListener("praxis:insert-prompt", handler);
  }, []);

  useEffect(() => {
    if (activeConvRef.current !== conversationId) {
      // Cancel any streaming poll tied to the conversation we're leaving so a
      // late tick can't write a stale answer into the freshly reset drafts.
      stopPoll();
      setSending(false);
      setDrafts([]);
      setError(null);
      activeConvRef.current = conversationId;
    }
  }, [conversationId]);

  // Cancel any in-flight poll on unmount.
  useEffect(() => stopPoll, []);

  const displayed = useMemo<DraftMessage[]>(() => {
    const base: DraftMessage[] = initialMessages.map((m) => ({
      id: m.messageId,
      role: m.role === "assistant" ? "assistant" : "user",
      content: m.content,
      // Only assistant messages carry sources on the server; the optional
      // chaining here keeps user/system rows clean of an empty array.
      sources: m.role === "assistant" ? m.sources : undefined,
    }));
    return base.concat(drafts);
  }, [initialMessages, drafts]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [displayed.length]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || sending) return;

    const pendingId = `pending-${Date.now()}`;
    const assistantId = `${pendingId}-assistant`;

    setDrafts((prev) => [
      ...prev,
      { id: pendingId, role: "user", content: trimmed },
      { id: assistantId, role: "assistant", content: "", pending: true },
    ]);
    setInput("");
    setSending(true);
    setError(null);

    try {
      // 1) Kick off the background worker. Returns the coordinates we poll.
      const start = await postChatStream(accessToken, {
        message: trimmed,
        conversationId: conversationId ?? undefined,
        model,
      });

      // Surface the new conversation immediately so the sidebar/url update
      // while the answer is still streaming.
      if (!conversationId) {
        onConversationCreated(start.conversationId, trimmed.slice(0, 80));
      }

      // 2) Poll until the message status is terminal. Stop on unmount /
      //    conversation switch (stopPoll) and after a safety cap (~16 min).
      const POLL_INTERVAL_MS = 800;
      const MAX_POLLS = 1200; // ~16 min at 800ms
      let polls = 0;

      await new Promise<void>((resolve) => {
        // Cancel any earlier poll before starting a new one (defensive — a
        // previous send should already be done since the composer is locked).
        stopPoll();
        const controller = { handle: 0, cancelled: false, settle: resolve };

        const finish = () => {
          if (controller.cancelled) return;
          stopPoll();
          resolve();
        };

        const tick = async () => {
          if (controller.cancelled) return;
          polls += 1;

          if (polls > MAX_POLLS) {
            setError("The response timed out. Please try again.");
            setDrafts((prev) =>
              prev
                .map((d) => (d.id === assistantId ? { ...d, pending: false } : d))
                .filter((d) => !(d.id === assistantId && !d.content)),
            );
            finish();
            return;
          }

          let poll;
          try {
            poll = await pollChatStream(
              accessToken,
              start.conversationId,
              start.sortKey,
            );
          } catch {
            // Swallow transient poll errors — the next tick retries. A
            // persistent failure eventually trips the MAX_POLLS safety cap.
            return;
          }
          if (controller.cancelled) return;

          if (poll.status === "complete") {
            setDrafts((prev) =>
              prev.map((d) =>
                d.id === assistantId
                  ? {
                      ...d,
                      content: poll.content,
                      pending: false,
                      sources: poll.sources,
                    }
                  : d,
              ),
            );
            finish();
            return;
          }

          if (poll.status === "error") {
            setError("The assistant ran into a problem generating a response.");
            setDrafts((prev) =>
              prev
                .map((d) =>
                  d.id === assistantId
                    ? { ...d, content: poll.content, pending: false }
                    : d,
                )
                // Drop the placeholder entirely if nothing streamed back, to
                // match the existing catch-block UX.
                .filter((d) => !(d.id === assistantId && !poll.content)),
            );
            finish();
            return;
          }

          // status === "streaming": fold the partial text in, stay pending.
          setDrafts((prev) =>
            prev.map((d) =>
              d.id === assistantId ? { ...d, content: poll.content } : d,
            ),
          );
        };

        controller.handle = window.setInterval(() => {
          void tick();
        }, POLL_INTERVAL_MS);
        pollRef.current = controller;
        // Fire an immediate first poll so a fast answer doesn't wait a full
        // interval before showing anything.
        void tick();
      });
    } catch (err) {
      // Kickoff (postChatStream) failed — mirror the original catch UX.
      stopPoll();
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      setDrafts((prev) => prev.filter((d) => d.id !== assistantId));
    } finally {
      setSending(false);
    }
  }

  const handleTranslateSubmit = async (
    attachmentId: string,
    targetLanguage: TranslateLanguage,
  ) => {
    return createTranslateJob({ attachmentId, targetLanguage });
  };

  return (
    <section className="chat">
      <div className="chat__toolbar">
        <ModelPicker value={model} onChange={setModel} disabled={sending} />
      </div>
      {translateJobs.length > 0 && (
        <div className="chat__job-cards">
          {translateJobs.map((j) => (
            <JobCard
              key={j.jobId}
              job={j}
              accessToken={accessToken}
              onJobUpdated={updateTranslateJob}
              onDismiss={() => dismissTranslateJob(j.jobId)}
            />
          ))}
        </div>
      )}
      <div className="chat__messages">
        {loading && <MessagesSkeleton />}
        {!loading && displayed.length === 0 && (
          <div className="chat__empty">
            <img src="/anna_logo.png" alt="" className="chat__empty-logo" />
            <h2 className="chat__empty-title">How can Praxis help today?</h2>
            <p className="chat__empty-body">
              Ask a clinical question, analyze a treatment plan, or draft a note.
              All conversations stay inside ANNA's HIPAA-covered AWS environment.
            </p>
          </div>
        )}
        {displayed.map((m) => (
          <Message
            key={m.id}
            role={m.role}
            content={m.content}
            pending={m.pending}
            sources={m.sources}
          />
        ))}
        <div ref={bottomRef} />
      </div>

      {error && (
        <div className="chat__error">
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)} className="btn btn--ghost">
            Dismiss
          </button>
        </div>
      )}

      {attachments.length > 0 && (
        <div className="chat__attachments">
          {attachments.map((a) => (
            <AttachmentChip
              key={a.attachmentId}
              attachment={a}
              onRemove={removeAttachment}
              onTranslate={() => setTranslateTarget(a)}
            />
          ))}
        </div>
      )}

      {translateTarget && (
        <TranslateDialog
          open={translateTarget !== null}
          onClose={() => setTranslateTarget(null)}
          attachment={translateTarget}
          accessToken={accessToken}
          onSubmit={handleTranslateSubmit}
          onJobCreated={() => {
            // Hook already appended the JobCard; nothing else to do here.
          }}
          // The "View past translations →" link routes through a global
          // event so ChatPage (which owns the RecentTranslations modal,
          // alongside other library modals like KnowledgeBase) can open
          // it without prop-drilling.
          onOpenRecent={() => {
            window.dispatchEvent(new Event("praxis:open-recent-translations"));
          }}
        />
      )}

      <form
        className="chat__composer chat__composer--with-picker"
        onSubmit={handleSubmit}
      >
        <div className="chat__composer-picker" title="Attach files">
          <AttachmentPicker
            onFiles={(files) => void uploadFiles(files)}
            disabled={sending}
          />
        </div>
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Message Praxis…"
          rows={3}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void handleSubmit(e);
            }
          }}
          disabled={sending}
        />
        <button type="submit" className="btn btn--primary" disabled={sending || !input.trim()}>
          {sending ? "Sending…" : "Send"}
        </button>
      </form>
    </section>
  );
}

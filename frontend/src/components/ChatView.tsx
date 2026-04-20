import { useEffect, useMemo, useRef, useState } from "react";

import { postChat } from "../api/chat";
import type { MessageSummary } from "../api/conversations";
import { DEFAULT_MODEL, ModelPicker } from "./ModelPicker";
import { Message } from "./Message";
import { MessagesSkeleton } from "./Skeleton";

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
      setDrafts([]);
      setError(null);
      activeConvRef.current = conversationId;
    }
  }, [conversationId]);

  const displayed = useMemo<DraftMessage[]>(() => {
    const base: DraftMessage[] = initialMessages.map((m) => ({
      id: m.messageId,
      role: m.role === "assistant" ? "assistant" : "user",
      content: m.content,
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
      const resp = await postChat(accessToken, {
        message: trimmed,
        conversationId: conversationId ?? undefined,
        model,
      });

      setDrafts((prev) =>
        prev.map((d) =>
          d.id === assistantId
            ? { ...d, content: resp.assistantMessage, pending: false }
            : d,
        ),
      );

      if (!conversationId) {
        onConversationCreated(resp.conversationId, trimmed.slice(0, 80));
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      setDrafts((prev) => prev.filter((d) => d.id !== assistantId));
    } finally {
      setSending(false);
    }
  }

  return (
    <section className="chat">
      <div className="chat__toolbar">
        <ModelPicker value={model} onChange={setModel} disabled={sending} />
      </div>
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
          <Message key={m.id} role={m.role} content={m.content} pending={m.pending} />
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

      <form className="chat__composer" onSubmit={handleSubmit}>
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

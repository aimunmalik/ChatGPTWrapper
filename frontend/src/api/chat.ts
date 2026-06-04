import { apiFetch } from "./client";

export interface ChatRequest {
  message: string;
  conversationId?: string;
  model?: string;
}

/**
 * Per-source retrieval metadata returned with each chat response. The
 * assistant may cite these inline as `[1]`, `[2]`, etc. — `index` is the
 * number used in those citations. `score` is the raw cosine similarity; the
 * UI surfaces it only as a coarse hover tooltip.
 *
 * See docs/KB_CONTRACT.md for the backend retrieval contract.
 */
export interface Source {
  index: number;
  /** Points the download endpoint at the underlying PDF/DOCX/etc. */
  kbDocId: string;
  docTitle: string;
  sourceType: string;
  pageNumber?: number;
  score: number;
}

export interface ChatResponse {
  conversationId: string;
  messageId: string;
  assistantMessage: string;
  tokens: { input: number; output: number };
  model: string;
  /** Chunks retrieved + injected into the prompt, in citation order.
   *  `[]` when no chunks cleared the min_score threshold. */
  sources: Source[];
}

export function postChat(accessToken: string, req: ChatRequest): Promise<ChatResponse> {
  return apiFetch<ChatResponse>("/chat", accessToken, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

/**
 * Streaming chat (Path B — async worker + poll). See docs/STREAMING_CONTRACT.md.
 *
 * `postChatStream` kicks off generation: the backend persists the user message,
 * creates an empty assistant message, async-invokes the worker, and returns the
 * coordinates the client needs to poll. `pollChatStream` reads the partial /
 * final assistant text until `status` is terminal.
 */
export interface ChatStreamStart {
  conversationId: string;
  messageId: string;
  sortKey: string;
  status: string;
}

export interface ChatStreamPoll {
  conversationId: string;
  messageId: string;
  sortKey: string;
  status: "streaming" | "complete" | "error";
  content: string;
  sources: Source[];
  tokens: { input: number; output: number };
  model: string;
}

export function postChatStream(
  accessToken: string,
  req: ChatRequest,
): Promise<ChatStreamStart> {
  return apiFetch<ChatStreamStart>("/chat/stream", accessToken, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function pollChatStream(
  accessToken: string,
  conversationId: string,
  sortKey: string,
): Promise<ChatStreamPoll> {
  // sortKey contains `#`, so both query params are percent-encoded.
  const query = `?cid=${encodeURIComponent(conversationId)}&sk=${encodeURIComponent(sortKey)}`;
  return apiFetch<ChatStreamPoll>(`/chat/stream${query}`, accessToken, {
    method: "GET",
  });
}

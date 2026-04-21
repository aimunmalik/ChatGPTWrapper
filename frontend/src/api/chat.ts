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

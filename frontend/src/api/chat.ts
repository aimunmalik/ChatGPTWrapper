import { apiFetch } from "./client";

export interface ChatRequest {
  message: string;
  conversationId?: string;
  model?: string;
}

export interface ChatResponse {
  conversationId: string;
  messageId: string;
  assistantMessage: string;
  tokens: { input: number; output: number };
  model: string;
}

export function postChat(accessToken: string, req: ChatRequest): Promise<ChatResponse> {
  return apiFetch<ChatResponse>("/chat", accessToken, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

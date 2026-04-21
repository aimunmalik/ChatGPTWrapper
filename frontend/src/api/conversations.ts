import { apiFetch } from "./client";
import type { Source } from "./chat";

export interface Conversation {
  userId: string;
  conversationId: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  model: string;
}

export interface MessageSummary {
  messageId: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt: number;
  /** Citations persisted alongside the assistant message. Absent on user/
   *  system messages and on assistant messages that predate KB retrieval. */
  sources?: Source[];
}

export interface ConversationDetail {
  conversation: Conversation;
  messages: MessageSummary[];
}

export function listConversations(
  accessToken: string,
): Promise<{ conversations: Conversation[] }> {
  return apiFetch("/conversations", accessToken);
}

export function getConversationMessages(
  accessToken: string,
  conversationId: string,
): Promise<ConversationDetail> {
  return apiFetch(
    `/conversations/${encodeURIComponent(conversationId)}/messages`,
    accessToken,
  );
}

export function deleteConversation(
  accessToken: string,
  conversationId: string,
): Promise<{ deleted: boolean }> {
  return apiFetch(`/conversations/${encodeURIComponent(conversationId)}`, accessToken, {
    method: "DELETE",
  });
}

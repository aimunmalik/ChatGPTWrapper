import { apiFetch } from "./client";

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

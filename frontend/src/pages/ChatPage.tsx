import { useEffect, useState } from "react";
import { useAuth } from "react-oidc-context";

import { ChatView } from "../components/ChatView";
import { Layout } from "../components/Layout";
import { Sidebar } from "../components/Sidebar";
import type { Conversation, MessageSummary } from "../api/conversations";
import {
  getConversationMessages,
  listConversations,
} from "../api/conversations";

export function ChatPage() {
  const auth = useAuth();
  const accessToken = auth.user?.access_token ?? "";

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<MessageSummary[]>([]);
  const [loadingList, setLoadingList] = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [listError, setListError] = useState<string | null>(null);

  useEffect(() => {
    if (!accessToken) return;
    let cancelled = false;
    setLoadingList(true);
    setListError(null);
    listConversations(accessToken)
      .then((resp) => {
        if (!cancelled) {
          setConversations(resp.conversations);
        }
      })
      .catch((err) => {
        if (!cancelled) setListError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoadingList(false);
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken]);

  useEffect(() => {
    if (!accessToken || !activeId) {
      setMessages([]);
      return;
    }
    let cancelled = false;
    setLoadingMessages(true);
    getConversationMessages(accessToken, activeId)
      .then((resp) => {
        if (!cancelled) {
          setMessages(resp.messages);
        }
      })
      .catch(() => {
        if (!cancelled) setMessages([]);
      })
      .finally(() => {
        if (!cancelled) setLoadingMessages(false);
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken, activeId]);

  function handleConversationCreated(convId: string, title: string) {
    setActiveId(convId);
    setConversations((prev) => {
      if (prev.some((c) => c.conversationId === convId)) return prev;
      const now = Date.now();
      const newConv: Conversation = {
        userId: auth.user?.profile.sub ?? "",
        conversationId: convId,
        title,
        createdAt: now,
        updatedAt: now,
        model: "",
      };
      return [newConv, ...prev];
    });
  }

  function handleConversationDeleted(convId: string) {
    setConversations((prev) =>
      prev.filter((c) => c.conversationId !== convId),
    );
    if (activeId === convId) {
      setActiveId(null);
      setMessages([]);
    }
  }

  return (
    <Layout>
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        loading={loadingList}
        error={listError}
        onSelect={(id) => setActiveId(id)}
        onNewChat={() => {
          setActiveId(null);
          setMessages([]);
        }}
        onDelete={handleConversationDeleted}
        accessToken={accessToken}
      />
      <ChatView
        conversationId={activeId}
        initialMessages={messages}
        loading={loadingMessages}
        accessToken={accessToken}
        onConversationCreated={handleConversationCreated}
      />
    </Layout>
  );
}

import { useCallback, useEffect, useMemo, useState } from "react";
import { useAuth } from "react-oidc-context";

import { ChatView } from "../components/ChatView";
import type { Command } from "../components/CommandPalette";
import { CommandPalette } from "../components/CommandPalette";
import { Layout } from "../components/Layout";
import { MODEL_OPTIONS } from "../components/ModelPicker";
import { PromptLibrary } from "../components/PromptLibrary";
import { Sidebar } from "../components/Sidebar";
import type { Conversation, MessageSummary } from "../api/conversations";
import {
  getConversationMessages,
  listConversations,
} from "../api/conversations";
import { buildLogoutUrl } from "../auth/oidcConfig";
import { usePrompts } from "../hooks/usePrompts";
import { PROMPT_TEMPLATES } from "../prompts";
import { useTheme } from "../theme/ThemeContext";

const MODEL_STORAGE_KEY = "anna-chat:model";

export function ChatPage() {
  const auth = useAuth();
  const accessToken = auth.user?.access_token ?? "";
  const theme = useTheme();

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<MessageSummary[]>([]);
  const [loadingList, setLoadingList] = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [promptLibraryOpen, setPromptLibraryOpen] = useState(false);

  const {
    prompts: userPrompts,
    create: createUserPrompt,
    update: updateUserPrompt,
    remove: removeUserPrompt,
  } = usePrompts({ accessToken });

  useEffect(() => {
    if (!accessToken) return;
    let cancelled = false;
    setLoadingList(true);
    setListError(null);
    listConversations(accessToken)
      .then((resp) => {
        if (!cancelled) setConversations(resp.conversations);
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
        if (!cancelled) setMessages(resp.messages);
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

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const isMod = e.metaKey || e.ctrlKey;
      if (isMod && (e.key === "k" || e.key === "K")) {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const handleNewChat = useCallback(() => {
    setActiveId(null);
    setMessages([]);
  }, []);

  const handleSignOut = useCallback(() => {
    const idToken = auth.user?.id_token;
    void auth.removeUser().finally(() => {
      window.location.href = buildLogoutUrl(idToken);
    });
  }, [auth]);

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
    setConversations((prev) => prev.filter((c) => c.conversationId !== convId));
    if (activeId === convId) {
      setActiveId(null);
      setMessages([]);
    }
  }

  const commands = useMemo<Command[]>(() => {
    const actions: Command[] = [
      {
        id: "new-chat",
        label: "New chat",
        group: "Actions",
        hint: "Start a fresh conversation",
        onRun: handleNewChat,
      },
      {
        id: "toggle-theme",
        label: theme.resolved === "dark" ? "Switch to light mode" : "Switch to dark mode",
        group: "Appearance",
        keywords: "dark light theme mode",
        onRun: theme.toggle,
      },
      {
        id: "sign-out",
        label: "Sign out",
        group: "Actions",
        hint: "End your session",
        keywords: "logout exit",
        onRun: handleSignOut,
      },
    ];

    const models: Command[] = MODEL_OPTIONS.map((m) => ({
      id: `model-${m.id}`,
      label: `Use ${m.label}`,
      group: "Model",
      hint: m.hint,
      onRun: () => {
        window.localStorage.setItem(MODEL_STORAGE_KEY, m.id);
        window.dispatchEvent(new Event("praxis:model-changed"));
      },
    }));

    const prompts: Command[] = PROMPT_TEMPLATES.map((p) => ({
      id: `prompt-${p.id}`,
      label: p.label,
      group: "Quick prompts",
      hint: p.hint,
      keywords: p.hint,
      onRun: () => {
        handleNewChat();
        window.dispatchEvent(
          new CustomEvent("praxis:insert-prompt", { detail: { text: p.template } }),
        );
      },
    }));

    const myPromptCmds: Command[] = userPrompts.map((p) => ({
      id: `mine-${p.promptId}`,
      label: p.title,
      group: "My prompts",
      keywords: p.body.slice(0, 200),
      onRun: () => {
        handleNewChat();
        window.dispatchEvent(
          new CustomEvent("praxis:insert-prompt", { detail: { text: p.body } }),
        );
      },
    }));

    const libraryCmds: Command[] = [
      {
        id: "manage-prompts",
        label: "Manage my prompts…",
        group: "Library",
        hint: "Create, edit, or delete your saved prompts",
        keywords: "prompts library manage",
        onRun: () => setPromptLibraryOpen(true),
      },
    ];

    const conversationCmds: Command[] = conversations.slice(0, 20).map((c) => ({
      id: `conv-${c.conversationId}`,
      label: c.title,
      group: "Open conversation",
      onRun: () => setActiveId(c.conversationId),
    }));

    return [
      ...actions,
      ...myPromptCmds,
      ...prompts,
      ...libraryCmds,
      ...models,
      ...conversationCmds,
    ];
  }, [conversations, handleNewChat, handleSignOut, theme, userPrompts]);

  return (
    <>
      <Layout onOpenCommandPalette={() => setPaletteOpen(true)}>
        <Sidebar
          conversations={conversations}
          activeId={activeId}
          loading={loadingList}
          error={listError}
          onSelect={(id) => setActiveId(id)}
          onNewChat={handleNewChat}
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
      <CommandPalette
        open={paletteOpen}
        commands={commands}
        onClose={() => setPaletteOpen(false)}
      />
      <PromptLibrary
        open={promptLibraryOpen}
        onClose={() => setPromptLibraryOpen(false)}
        prompts={userPrompts}
        onCreate={createUserPrompt}
        onUpdate={updateUserPrompt}
        onDelete={removeUserPrompt}
        starterTemplates={PROMPT_TEMPLATES}
      />
    </>
  );
}

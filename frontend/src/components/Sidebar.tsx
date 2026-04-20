import { useState } from "react";
import clsx from "clsx";

import type { Conversation } from "../api/conversations";
import { deleteConversation } from "../api/conversations";
import { SidebarSkeleton } from "./Skeleton";

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  loading: boolean;
  error: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  onDelete: (id: string) => void;
  accessToken: string;
}

export function Sidebar({
  conversations,
  activeId,
  loading,
  error,
  onSelect,
  onNewChat,
  onDelete,
  accessToken,
}: Props) {
  const [deletingId, setDeletingId] = useState<string | null>(null);

  async function handleDelete(id: string, title: string) {
    if (!confirm(`Delete conversation "${title}"? This cannot be undone.`)) return;
    setDeletingId(id);
    try {
      await deleteConversation(accessToken, id);
      onDelete(id);
    } catch (err) {
      alert(`Failed to delete: ${(err as Error).message}`);
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <aside className="sidebar">
      <button type="button" className="btn btn--primary sidebar__new" onClick={onNewChat}>
        + New chat
      </button>

      {loading && <SidebarSkeleton />}
      {error && <div className="sidebar__status error">{error}</div>}

      <ul className="sidebar__list">
        {conversations.map((c) => (
          <li
            key={c.conversationId}
            className={clsx("sidebar__item", {
              "sidebar__item--active": activeId === c.conversationId,
            })}
          >
            <button
              type="button"
              className="sidebar__item-title"
              onClick={() => onSelect(c.conversationId)}
              title={c.title}
            >
              {c.title}
            </button>
            <button
              type="button"
              className="sidebar__item-delete"
              onClick={() => handleDelete(c.conversationId, c.title)}
              disabled={deletingId === c.conversationId}
              aria-label={`Delete ${c.title}`}
            >
              ×
            </button>
          </li>
        ))}
      </ul>

      {!loading && conversations.length === 0 && !error && (
        <div className="sidebar__empty">No conversations yet.<br/>Start one on the right.</div>
      )}

      <div className="sidebar__hint">
        <span>Quick search</span>
        <span>
          <span className="kbd">⌘</span> <span className="kbd">K</span>
        </span>
      </div>
    </aside>
  );
}

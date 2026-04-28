import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";

import type { AdminUser, IdentitySource } from "../api/admin";
import type { RowError } from "../hooks/useAdminUsers";
import { useAdminUsers } from "../hooks/useAdminUsers";

interface Props {
  open: boolean;
  onClose: () => void;
  accessToken: string;
}

export function AdminUsers({ open, onClose, accessToken }: Props) {
  const {
    users,
    isLoading,
    error,
    rowErrors,
    invite,
    update,
    signOut,
    clearRowError,
  } = useAdminUsers({ accessToken });

  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteName, setInviteName] = useState("");
  const [inviteAdmin, setInviteAdmin] = useState(false);
  const [inviting, setInviting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [inviteToast, setInviteToast] = useState<string | null>(null);

  // Reset the invite form whenever the modal closes so the next open is a
  // clean slate. Mirrors the KnowledgeBase reset pattern.
  useEffect(() => {
    if (!open) {
      setInviteEmail("");
      setInviteName("");
      setInviteAdmin(false);
      setFormError(null);
      setInviteToast(null);
    }
  }, [open]);

  // Esc closes — no in-flight gating needed; mutations are quick API calls
  // and rolling back mid-flight is safe.
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  async function handleInvite(e: React.FormEvent) {
    e.preventDefault();
    setFormError(null);
    setInviteToast(null);

    const email = inviteEmail.trim();
    const name = inviteName.trim();
    if (!email) {
      setFormError("Email is required.");
      return;
    }
    if (!name) {
      setFormError("Name is required.");
      return;
    }

    setInviting(true);
    try {
      const created = await invite({ email, name, isAdmin: inviteAdmin });
      if (created) {
        setInviteToast(`Welcome email sent to ${created.email || email}.`);
        setInviteEmail("");
        setInviteName("");
        setInviteAdmin(false);
      }
    } finally {
      setInviting(false);
    }
  }

  // Sort: admins first, then enabled, then name. Stable enough that toggling
  // a user's admin flag doesn't surprise-jump them off-screen during the
  // optimistic update window.
  const sorted = useMemo(() => {
    return [...users].sort((a, b) => {
      if (a.isAdmin !== b.isAdmin) return a.isAdmin ? -1 : 1;
      if (a.enabled !== b.enabled) return a.enabled ? -1 : 1;
      return (a.name || a.email).localeCompare(b.name || b.email);
    });
  }, [users]);

  if (!open) return null;

  return (
    <div
      className="cmdk-overlay admin-users"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="cmdk-panel admin-users__panel"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Manage users"
        aria-modal="true"
      >
        <div className="admin-users__header">
          <h2 className="admin-users__title">Manage users</h2>
          <button
            type="button"
            className="admin-users__close"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div className="admin-users__body">
          <form className="admin-users__invite" onSubmit={handleInvite}>
            <div className="admin-users__invite-title">Invite a user</div>
            <div className="admin-users__invite-row">
              <input
                type="email"
                className="admin-users__input"
                placeholder="name@example.com"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
                disabled={inviting}
                autoComplete="off"
                aria-label="Email address"
              />
              <input
                type="text"
                className="admin-users__input"
                placeholder="Display name"
                value={inviteName}
                onChange={(e) => setInviteName(e.target.value)}
                disabled={inviting}
                autoComplete="off"
                aria-label="Display name"
              />
              <label className="admin-users__checkbox">
                <input
                  type="checkbox"
                  checked={inviteAdmin}
                  onChange={(e) => setInviteAdmin(e.target.checked)}
                  disabled={inviting}
                />
                <span>Admin</span>
              </label>
              <button
                type="submit"
                className="btn btn--primary admin-users__invite-submit"
                disabled={inviting}
              >
                {inviting ? "Sending…" : "Send invite"}
              </button>
            </div>
            {formError && (
              <div className="admin-users__error">{formError}</div>
            )}
            {inviteToast && (
              <div className="admin-users__toast">{inviteToast}</div>
            )}
          </form>

          <div className="admin-users__list-header">
            <h3 className="admin-users__list-title">Users</h3>
            <span className="admin-users__list-count">{users.length}</span>
          </div>

          {isLoading && users.length === 0 ? (
            <div className="admin-users__empty">Loading…</div>
          ) : error && users.length === 0 ? (
            <div className="admin-users__error">{error}</div>
          ) : users.length === 0 ? (
            <div className="admin-users__empty">No users in the pool.</div>
          ) : (
            <ul className="admin-users__list">
              {sorted.map((u) => (
                <UserRow
                  key={u.username}
                  user={u}
                  rowError={rowErrors[u.username]}
                  onUpdate={update}
                  onSignOut={signOut}
                  onClearError={clearRowError}
                />
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

interface RowProps {
  user: AdminUser;
  rowError: RowError | undefined;
  onUpdate: (
    username: string,
    req: { enabled?: boolean; isAdmin?: boolean },
  ) => Promise<void>;
  onSignOut: (username: string) => Promise<void>;
  onClearError: (username: string) => void;
}

function UserRow({
  user,
  rowError,
  onUpdate,
  onSignOut,
  onClearError,
}: RowProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  // Close popover on outside click. Keyboard escape is handled by the
  // modal-level listener.
  useEffect(() => {
    if (!menuOpen) return;
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    window.addEventListener("mousedown", handler);
    return () => window.removeEventListener("mousedown", handler);
  }, [menuOpen]);

  function runAndClose(action: () => Promise<void>) {
    setMenuOpen(false);
    void action();
  }

  return (
    <li
      className={clsx(
        "admin-users__item",
        !user.enabled && "admin-users__item--disabled",
      )}
    >
      <div className="admin-users__item-row">
        <div className="admin-users__item-main">
          <span className="admin-users__item-name" title={user.name}>
            {user.name || "—"}
          </span>
          <span className="admin-users__item-email" title={user.email}>
            {user.email || "(no email)"}
          </span>
        </div>
        <div className="admin-users__item-meta">
          <IdentityPill source={user.identitySource} />
          {user.isAdmin && (
            <span
              className="admin-users__pill admin-users__pill--admin"
              title="In the admins group"
            >
              Admin
            </span>
          )}
          <StatusPill enabled={user.enabled} />
        </div>
        <div className="admin-users__item-actions" ref={menuRef}>
          <button
            type="button"
            className="admin-users__menu-btn"
            onClick={() => setMenuOpen((o) => !o)}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            aria-label={`Actions for ${user.name || user.email}`}
          >
            ⋯
          </button>
          {menuOpen && (
            <div className="admin-users__menu" role="menu">
              <button
                type="button"
                className="admin-users__menu-item"
                role="menuitem"
                onClick={() =>
                  runAndClose(() =>
                    onUpdate(user.username, { isAdmin: !user.isAdmin }),
                  )
                }
              >
                {user.isAdmin ? "Remove admin" : "Make admin"}
              </button>
              <button
                type="button"
                className="admin-users__menu-item"
                role="menuitem"
                onClick={() =>
                  runAndClose(() =>
                    onUpdate(user.username, { enabled: !user.enabled }),
                  )
                }
              >
                {user.enabled ? "Disable" : "Enable"}
              </button>
              <button
                type="button"
                className="admin-users__menu-item"
                role="menuitem"
                onClick={() => runAndClose(() => onSignOut(user.username))}
              >
                Sign out everywhere
              </button>
            </div>
          )}
        </div>
      </div>
      {rowError && (
        <div className="admin-users__row-error" role="alert">
          <span>{rowError.message}</span>
          <button
            type="button"
            className="admin-users__row-error-dismiss"
            onClick={() => onClearError(user.username)}
            aria-label="Dismiss error"
          >
            ×
          </button>
        </div>
      )}
    </li>
  );
}

function IdentityPill({ source }: { source: IdentitySource }) {
  const label = source === "microsoft" ? "Microsoft" : "Local";
  return (
    <span
      className={clsx(
        "admin-users__pill",
        `admin-users__pill--source-${source}`,
      )}
      title={
        source === "microsoft"
          ? "Federated via Microsoft 365 SSO"
          : "Local Cognito user"
      }
    >
      {label}
    </span>
  );
}

function StatusPill({ enabled }: { enabled: boolean }) {
  if (enabled) {
    return (
      <span
        className="admin-users__pill admin-users__pill--enabled"
        title="Account enabled"
        aria-label="Enabled"
      >
        ●
      </span>
    );
  }
  return (
    <span
      className="admin-users__pill admin-users__pill--off"
      title="Account disabled"
    >
      ⊘ Disabled
    </span>
  );
}

import { useCallback, useEffect, useRef, useState } from "react";

import {
  inviteAdminUser,
  listAdminUsers,
  signOutAdminUser,
  updateAdminUser,
} from "../api/admin";
import type {
  AdminErrorBody,
  AdminErrorType,
  AdminUser,
  InviteUserRequest,
  UpdateUserRequest,
} from "../api/admin";
import type { ApiError } from "../api/client";

interface Options {
  accessToken: string;
}

/**
 * Per-row error surfaced to the UI after a failed mutation. Carries the
 * `errorType` discriminator so the modal can render specific wording for
 * SelfDisable / SelfDemote / LastAdmin instead of a generic "Failed".
 */
export interface RowError {
  message: string;
  errorType?: AdminErrorType;
}

interface UseAdminUsers {
  users: AdminUser[];
  isLoading: boolean;
  /** Top-level load/list error (e.g. initial GET failed). Per-row mutation
   *  errors live in `rowErrors`, keyed by username. */
  error: string | null;
  /** Per-row mutation errors. Cleared automatically before each new
   *  mutation against the same row. */
  rowErrors: Record<string, RowError | undefined>;
  invite: (req: InviteUserRequest) => Promise<AdminUser | null>;
  update: (username: string, req: UpdateUserRequest) => Promise<void>;
  signOut: (username: string) => Promise<void>;
  refresh: () => Promise<void>;
  /** Manually clear the inline error attached to one row. */
  clearRowError: (username: string) => void;
}

/** Best-effort extraction of the contract's `errorType` discriminator from
 *  an ApiError. Lives here (not in api/client) because the shape is admin-
 *  specific. Falls back to `undefined` so callers can still render a
 *  generic message. */
function readErrorBody(err: unknown): AdminErrorBody | null {
  if (!err || typeof err !== "object") return null;
  const apiErr = err as ApiError;
  if (!apiErr.body || typeof apiErr.body !== "object") return null;
  return apiErr.body as AdminErrorBody;
}

function toRowError(err: unknown, fallback: string): RowError {
  const body = readErrorBody(err);
  if (body?.errorType) {
    return {
      errorType: body.errorType,
      message: body.message ?? defaultMessageFor(body.errorType),
    };
  }
  if (body?.message) return { message: body.message };
  if (err instanceof Error) return { message: err.message || fallback };
  return { message: fallback };
}

/** Friendly per-rule wording the UI can fall back to when the backend
 *  doesn't include a `message` alongside `errorType`. */
function defaultMessageFor(t: AdminErrorType): string {
  switch (t) {
    case "SelfDisable":
      return "You can't disable your own account.";
    case "SelfDemote":
      return "You can't remove yourself from the admins group.";
    case "LastAdmin":
      return "Refusing to remove the last admin.";
  }
}

export function useAdminUsers({ accessToken }: Options): UseAdminUsers {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rowErrors, setRowErrors] = useState<
    Record<string, RowError | undefined>
  >({});

  const tokenRef = useRef(accessToken);
  useEffect(() => {
    tokenRef.current = accessToken;
  }, [accessToken]);

  const refresh = useCallback(async () => {
    const token = tokenRef.current;
    if (!token) {
      setUsers([]);
      return;
    }
    try {
      const resp = await listAdminUsers(token);
      setUsers(resp.users);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load users");
    }
  }, []);

  // Initial load whenever the access token changes — same lifecycle as
  // useKbDocuments so a remount-with-new-token resets cleanly.
  useEffect(() => {
    let cancelled = false;
    if (!accessToken) {
      setUsers([]);
      setIsLoading(false);
      return () => {
        cancelled = true;
      };
    }
    setIsLoading(true);
    setError(null);
    listAdminUsers(accessToken)
      .then((resp) => {
        if (!cancelled) setUsers(resp.users);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load users");
          setUsers([]);
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken]);

  const clearRowError = useCallback((username: string) => {
    setRowErrors((prev) => {
      if (!(username in prev)) return prev;
      const next = { ...prev };
      delete next[username];
      return next;
    });
  }, []);

  const invite = useCallback(
    async (req: InviteUserRequest): Promise<AdminUser | null> => {
      const token = tokenRef.current;
      if (!token) return null;
      setError(null);
      try {
        const created = await inviteAdminUser(token, req);
        // Optimistically prepend so the new row shows up immediately. The
        // backend returns the canonical shape so no follow-up refresh is
        // needed for accuracy.
        setUsers((prev) => {
          if (prev.some((u) => u.username === created.username)) return prev;
          return [created, ...prev];
        });
        return created;
      } catch (err) {
        // Invite errors aren't row-scoped (no row exists yet). Surface as
        // top-level error so the form can render it inline.
        const body = readErrorBody(err);
        setError(
          body?.message ??
            (err instanceof Error ? err.message : "Failed to invite user"),
        );
        return null;
      }
    },
    [],
  );

  const update = useCallback(
    async (username: string, req: UpdateUserRequest): Promise<void> => {
      const token = tokenRef.current;
      if (!token) return;

      // Snapshot for rollback. Capture by username, not index, so a
      // concurrent invite that prepends to the list doesn't corrupt the
      // restored state.
      const snapshot = users;
      const target = snapshot.find((u) => u.username === username);
      if (!target) return;

      // Clear any stale row error before the optimistic flip.
      clearRowError(username);

      // Optimistic patch — apply only the fields the caller specified.
      setUsers((prev) =>
        prev.map((u) =>
          u.username === username
            ? {
                ...u,
                ...(req.enabled !== undefined ? { enabled: req.enabled } : {}),
                ...(req.isAdmin !== undefined ? { isAdmin: req.isAdmin } : {}),
              }
            : u,
        ),
      );

      try {
        const updated = await updateAdminUser(token, username, req);
        // Replace with the server's canonical version (picks up
        // lastModifiedAt + any backend-side normalization).
        setUsers((prev) =>
          prev.map((u) => (u.username === username ? updated : u)),
        );
      } catch (err) {
        // Rollback to snapshot and attach the typed error to this row so
        // the modal can render SelfDisable / SelfDemote / LastAdmin
        // wording inline. Mirror useKbDocuments.remove rollback shape.
        setUsers(snapshot);
        setRowErrors((prev) => ({
          ...prev,
          [username]: toRowError(err, "Failed to update user"),
        }));
        // Resync silently in case the server diverged for a non-error
        // reason (e.g. a parallel admin made a change).
        await refresh();
      }
    },
    [users, refresh, clearRowError],
  );

  const signOut = useCallback(
    async (username: string): Promise<void> => {
      const token = tokenRef.current;
      if (!token) return;
      clearRowError(username);
      try {
        await signOutAdminUser(token, username);
        // No optimistic state change — sign-out doesn't affect any visible
        // field on the row. Surface success silently; caller can show a
        // toast if it wants. (KB delete pattern: action with no UI flip.)
      } catch (err) {
        setRowErrors((prev) => ({
          ...prev,
          [username]: toRowError(err, "Failed to sign user out"),
        }));
      }
    },
    [clearRowError],
  );

  return {
    users,
    isLoading,
    error,
    rowErrors,
    invite,
    update,
    signOut,
    refresh,
    clearRowError,
  };
}

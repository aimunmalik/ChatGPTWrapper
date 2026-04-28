import { apiFetch } from "./client";

/**
 * How the user's identity entered the Cognito pool. `microsoft` users were
 * federated via the M365 SSO IdP (their Cognito username is prefixed
 * `microsoft_…`). Everyone else is `local` — invited directly into the pool
 * with email + temporary password.
 *
 * Mirrors the union in docs/ADMIN_USERS_CONTRACT.md → "API surface".
 */
export type IdentitySource = "local" | "microsoft";

/**
 * Shape of one row in `GET /admin/users`. Mirrors the response example in
 * the contract — including the federated-user case where `email` may be
 * empty (`""`) when attribute mapping didn't populate it.
 */
export interface AdminUser {
  username: string;
  email: string;
  name: string;
  /** Cognito's raw status string. `EXTERNAL_PROVIDER` for federated users,
   *  otherwise things like `CONFIRMED`, `FORCE_CHANGE_PASSWORD`, etc. */
  status: string;
  enabled: boolean;
  isAdmin: boolean;
  identitySource: IdentitySource;
  createdAt: number;
  lastModifiedAt: number;
}

export interface InviteUserRequest {
  email: string;
  name: string;
  isAdmin: boolean;
}

/**
 * PATCH body for `/admin/users/{username}`. Both fields are optional —
 * the contract supports updating either or both in a single call. We
 * always send them together when a single UI action toggles both
 * (e.g. row optimistic updates that need atomic rollback).
 */
export interface UpdateUserRequest {
  enabled?: boolean;
  isAdmin?: boolean;
}

/** Discriminator strings the backend returns in the error body for the
 *  three self-protection rules. Surfaced verbatim so the UI can show a
 *  per-row inline message instead of a generic "Failed". */
export type AdminErrorType = "SelfDisable" | "SelfDemote" | "LastAdmin";

export interface AdminErrorBody {
  errorType?: AdminErrorType;
  message?: string;
}

export function listAdminUsers(
  accessToken: string,
): Promise<{ users: AdminUser[] }> {
  return apiFetch<{ users: AdminUser[] }>("/admin/users", accessToken);
}

export function inviteAdminUser(
  accessToken: string,
  req: InviteUserRequest,
): Promise<AdminUser> {
  return apiFetch<AdminUser>("/admin/users", accessToken, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function updateAdminUser(
  accessToken: string,
  username: string,
  req: UpdateUserRequest,
): Promise<AdminUser> {
  return apiFetch<AdminUser>(
    `/admin/users/${encodeURIComponent(username)}`,
    accessToken,
    {
      method: "PATCH",
      body: JSON.stringify(req),
    },
  );
}

export function signOutAdminUser(
  accessToken: string,
  username: string,
): Promise<{ username: string; signedOut: boolean }> {
  return apiFetch<{ username: string; signedOut: boolean }>(
    `/admin/users/${encodeURIComponent(username)}/sign-out`,
    accessToken,
    { method: "POST" },
  );
}

import { useAuth } from "react-oidc-context";

/**
 * Returns true iff the authenticated user is in the Cognito `admins` group.
 *
 * Cognito ID tokens include `cognito:groups` as a JSON array of strings when
 * the user belongs to one or more groups. When they belong to no groups the
 * claim is absent entirely — hence the defensive Array.isArray check.
 *
 * KB management UI + any future admin-only surfaces gate on this hook.
 */
export function useIsAdmin(): boolean {
  const { user } = useAuth();
  const groups = user?.profile["cognito:groups"] as string[] | undefined;
  return Array.isArray(groups) && groups.includes("admins");
}

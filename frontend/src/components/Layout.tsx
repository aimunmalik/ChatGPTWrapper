import { useAuth } from "react-oidc-context";

import { buildLogoutUrl } from "../auth/oidcConfig";

export function Layout({ children }: { children: React.ReactNode }) {
  const auth = useAuth();

  function handleSignOut() {
    const idToken = auth.user?.id_token;
    void auth.removeUser().finally(() => {
      window.location.href = buildLogoutUrl(idToken);
    });
  }

  return (
    <div className="layout">
      <header className="layout__header">
        <div className="layout__brand">
          <img src="/anna_logo.png" alt="" className="layout__logo" />
          <span>ANNA Chat</span>
        </div>
        <div className="layout__user">
          <span className="layout__user-email">
            {auth.user?.profile.email ?? ""}
          </span>
          <button type="button" onClick={handleSignOut} className="btn btn--ghost">
            Sign out
          </button>
        </div>
      </header>
      <main className="layout__main">{children}</main>
    </div>
  );
}

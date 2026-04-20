import { useAuth } from "react-oidc-context";

import { buildLogoutUrl } from "../auth/oidcConfig";
import { ThemeToggle } from "./ThemeToggle";

interface Props {
  children: React.ReactNode;
  onOpenCommandPalette?: () => void;
}

export function Layout({ children, onOpenCommandPalette }: Props) {
  const auth = useAuth();

  function handleSignOut() {
    const idToken = auth.user?.id_token;
    void auth.removeUser().finally(() => {
      window.location.href = buildLogoutUrl(idToken);
    });
  }

  const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

  return (
    <div className="layout">
      <header className="layout__header">
        <div className="layout__brand">
          <img src="/anna_logo.png" alt="" className="layout__logo" />
          <span className="layout__wordmark">Praxis</span>
          <span className="layout__by">by ANNA</span>
        </div>
        <div className="layout__user">
          {onOpenCommandPalette && (
            <button
              type="button"
              className="btn btn--ghost"
              onClick={onOpenCommandPalette}
              title="Open command palette"
            >
              <span style={{ fontSize: 13 }}>Search</span>
              <span className="kbd">{isMac ? "⌘" : "Ctrl"}</span>
              <span className="kbd">K</span>
            </button>
          )}
          <ThemeToggle />
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

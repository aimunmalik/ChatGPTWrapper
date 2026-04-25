import { useEffect, useState } from "react";
import { useAuth } from "react-oidc-context";
import { Navigate, useSearchParams } from "react-router-dom";

/**
 * Sign-in landing page.
 *
 * Two modes:
 *
 * 1. **Default** — the user hit /login organically (first visit, deep link,
 *    expired session, etc.). We auto-redirect into the Cognito hosted UI
 *    so they don't have to click an extra button.
 *
 * 2. **Post-signout** — the user just clicked Sign out. We arrive here with
 *    `?signedout=1` in the URL. Show an explicit confirmation + a Sign in
 *    button. We do NOT auto-redirect, because doing so silently bounces the
 *    user through Cognito → Microsoft → straight back into the app (their
 *    M365 session is still alive, so SSO re-authenticates with no prompt).
 *    The whole point of "sign out" is letting the user be signed out for at
 *    least one frame.
 */
export function LoginPage() {
  const auth = useAuth();
  const [params] = useSearchParams();
  const justSignedOut = params.get("signedout") === "1";
  const [signingIn, setSigningIn] = useState(false);

  useEffect(() => {
    // Auto-redirect ONLY when this isn't a post-signout landing.
    if (
      !justSignedOut &&
      !auth.isAuthenticated &&
      !auth.isLoading &&
      !auth.error
    ) {
      void auth.signinRedirect();
    }
  }, [auth, justSignedOut]);

  if (auth.isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  function handleSignIn() {
    setSigningIn(true);
    void auth.signinRedirect();
  }

  return (
    <div className="login-shell">
      <img src="/anna_logo.png" alt="ANNA" className="login-logo" />
      <h1 className="login-wordmark">Praxis</h1>
      <p className="login-tagline">Clinical intelligence, by ANNA</p>
      {justSignedOut ? (
        <>
          <p className="login-status">You're signed out.</p>
          <button
            type="button"
            className="btn btn--primary"
            onClick={handleSignIn}
            disabled={signingIn}
          >
            {signingIn ? "Redirecting…" : "Sign in"}
          </button>
        </>
      ) : (
        <p className="login-status">Redirecting you to sign in…</p>
      )}
      {auth.error && (
        <div className="error">
          <p>{auth.error.message}</p>
          <button onClick={() => void auth.signinRedirect()}>Try again</button>
        </div>
      )}
    </div>
  );
}

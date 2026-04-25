import { useEffect } from "react";
import { useAuth } from "react-oidc-context";
import { Navigate } from "react-router-dom";

/**
 * Sign-in landing page.
 *
 * Always auto-redirects unauthenticated users into the Cognito hosted UI —
 * no special-cases needed. The post-sign-out flow is handled by a separate
 * static page at /signed-out.html (NOT this component) so AuthProvider
 * never mounts there and the user's still-alive M365 session can't silently
 * re-authenticate them.
 */
export function LoginPage() {
  const auth = useAuth();

  useEffect(() => {
    if (!auth.isAuthenticated && !auth.isLoading && !auth.error) {
      void auth.signinRedirect();
    }
  }, [auth]);

  if (auth.isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  return (
    <div className="login-shell">
      <img src="/anna_logo.png" alt="ANNA" className="login-logo" />
      <h1 className="login-wordmark">Praxis</h1>
      <p className="login-tagline">Clinical intelligence, by ANNA</p>
      <p className="login-status">Redirecting you to sign in…</p>
      {auth.error && (
        <div className="error">
          <p>{auth.error.message}</p>
          <button onClick={() => void auth.signinRedirect()}>Try again</button>
        </div>
      )}
    </div>
  );
}

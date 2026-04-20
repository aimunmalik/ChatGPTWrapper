import { useEffect } from "react";
import { useAuth } from "react-oidc-context";
import { useNavigate } from "react-router-dom";

export function CallbackPage() {
  const auth = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (auth.isAuthenticated && !auth.isLoading) {
      navigate("/", { replace: true });
    }
  }, [auth.isAuthenticated, auth.isLoading, navigate]);

  if (auth.error) {
    return (
      <div className="full-screen-status error">
        <p>Sign-in failed: {auth.error.message}</p>
        <button onClick={() => navigate("/login", { replace: true })}>Try again</button>
      </div>
    );
  }

  return <div className="full-screen-status">Completing sign-in…</div>;
}

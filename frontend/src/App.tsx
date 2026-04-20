import { useAuth } from "react-oidc-context";
import { Navigate, Route, Routes } from "react-router-dom";

import { CallbackPage } from "./pages/CallbackPage";
import { ChatPage } from "./pages/ChatPage";
import { LoginPage } from "./pages/LoginPage";

function RequireAuth({ children }: { children: React.ReactNode }) {
  const auth = useAuth();

  if (auth.isLoading) {
    return <div className="full-screen-status">Loading…</div>;
  }
  if (auth.error) {
    return (
      <div className="full-screen-status error">
        <p>Sign-in error: {auth.error.message}</p>
        <button onClick={() => void auth.signinRedirect()}>Try again</button>
      </div>
    );
  }
  if (!auth.isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  return <>{children}</>;
}

export function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/"
        element={
          <RequireAuth>
            <ChatPage />
          </RequireAuth>
        }
      />
      <Route path="/callback" element={<CallbackPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

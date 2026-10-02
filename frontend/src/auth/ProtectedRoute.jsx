import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "./AuthContext";

export function ProtectedRoute({ children }) {
  const auth = useAuth();
  const location = useLocation();

  if (auth.status === "loading") {
    return <div className="auth-state">Checking your session...</div>;
  }
  if (auth.status === "unavailable") {
    return <div className="auth-state"><strong>Authentication unavailable</strong><span>{auth.error?.message || "The backend cannot be reached."}</span><button className="button" onClick={auth.refreshUser}>Retry</button></div>;
  }
  if (!auth.authenticated) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }
  return children;
}

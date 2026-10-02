import { createContext, useContext, useEffect, useRef, useState } from "react";
import { authApi } from "../api/authApi";
import { setUnauthorizedHandler } from "../api/client";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [state, setState] = useState({ status: "loading", user: null, error: null });
  const initialized = useRef(false);

  useEffect(() => {
    if (initialized.current) return undefined;
    initialized.current = true;
    setUnauthorizedHandler(() => setState({ status: "unauthenticated", user: null, error: null }));
    authApi.me()
      .then((user) => setState({ status: "authenticated", user, error: null }))
      .catch((error) => {
        setState({
          status: error.status === 0 || error.status >= 500 ? "unavailable" : "unauthenticated",
          user: null,
          error,
        });
      });
    return () => setUnauthorizedHandler(null);
  }, []);

  async function login(email, password) {
    const result = await authApi.login({ email, password });
    setState({ status: "authenticated", user: result.user, error: null });
    return result.user;
  }

  async function signup(name, email, password) {
    const result = await authApi.signup({ name, email, password });
    setState({ status: "authenticated", user: result.user, error: null });
    return result.user;
  }

  async function logout() {
    try {
      await authApi.logout();
    } finally {
      setState({ status: "unauthenticated", user: null, error: null });
    }
  }

  async function refreshUser() {
    setState((current) => ({ ...current, status: "loading", error: null }));
    try {
      const user = await authApi.me();
      setState({ status: "authenticated", user, error: null });
      return user;
    } catch (error) {
      setState({
        status: error.status === 0 || error.status >= 500 ? "unavailable" : "unauthenticated",
        user: null,
        error,
      });
      throw error;
    }
  }

  return <AuthContext.Provider value={{ ...state, authenticated: state.status === "authenticated", login, signup, logout, refreshUser }}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}

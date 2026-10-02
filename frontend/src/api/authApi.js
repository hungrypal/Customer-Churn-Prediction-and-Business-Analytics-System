import { request } from "./client";

export const authApi = {
  signup: (payload) => request("/api/v1/auth/signup", {
    method: "POST",
    body: JSON.stringify(payload),
    skipUnauthorized: true,
  }),
  login: (payload) => request("/api/v1/auth/login", {
    method: "POST",
    body: JSON.stringify(payload),
    skipUnauthorized: true,
  }),
  me: () => request("/api/v1/auth/me", { skipUnauthorized: true }),
  logout: () => request("/api/v1/auth/logout", { method: "POST" }),
};

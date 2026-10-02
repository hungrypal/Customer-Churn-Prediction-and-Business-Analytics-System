const API_URL = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
let unauthorizedHandler = null;

export function setUnauthorizedHandler(handler) {
  unauthorizedHandler = handler;
}

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

export async function request(path, options = {}) {
  if (!API_URL) throw new ApiError("Backend unavailable: VITE_API_URL is not configured.", 0);
  const { skipUnauthorized = false, ...fetchOptions } = options;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch(`${API_URL}${path}`, {
      ...fetchOptions, signal: controller.signal, credentials: "include",
      headers: { "Content-Type": "application/json", ...fetchOptions.headers },
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = Array.isArray(body.detail)
        ? body.detail.map((item) => item.msg || "Invalid request").join(" ")
        : body.detail;
      throw new ApiError(detail || "The API request failed.", response.status);
    }
    return body;
  } catch (error) {
    if (error.status === 401 && !skipUnauthorized && unauthorizedHandler) unauthorizedHandler();
    if (error.name === "AbortError") throw new ApiError("Backend request timed out.", 0);
    throw error;
  } finally { window.clearTimeout(timeout); }
}

export const coreApi = { health: () => request("/health"), models: () => request("/api/v1/models") };

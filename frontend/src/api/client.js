const API_URL = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

export async function request(path, options = {}) {
  if (!API_URL) throw new ApiError("Backend unavailable: VITE_API_URL is not configured.", 0);
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch(`${API_URL}${path}`, {
      ...options, signal: controller.signal,
      headers: { "Content-Type": "application/json", ...options.headers },
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new ApiError(body.detail || "The API request failed.", response.status);
    return body;
  } catch (error) {
    if (error.name === "AbortError") throw new ApiError("Backend request timed out.", 0);
    throw error;
  } finally { window.clearTimeout(timeout); }
}

export const coreApi = { health: () => request("/health"), models: () => request("/api/v1/models") };

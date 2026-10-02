import { request } from "./client";
export const dashboardApi = {
  summary: () => request("/api/v1/dashboard"), preview: () => request("/api/v1/dashboard/dataset-preview"),
  performance: () => request("/api/v1/dashboard/model-performance"), matrices: () => request("/api/v1/dashboard/confusion-matrices"),
  importance: () => request("/api/v1/dashboard/feature-importance"),
};

import { request } from "./client";
export const analyticsApi = {
  summary: () => request("/api/v1/analytics"), riskDistribution: () => request("/api/v1/analytics/risk-distribution"),
  comparison: () => request("/api/v1/analytics/model-comparison"), importance: () => request("/api/v1/analytics/feature-importance"),
};

import { request } from "./client";
export const historyApi = { list: (page = 1, pageSize = 10) => request(`/api/v1/predictions/history?page=${page}&page_size=${pageSize}`), today: () => request("/api/v1/predictions/today") };

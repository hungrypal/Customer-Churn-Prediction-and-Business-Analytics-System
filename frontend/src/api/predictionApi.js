import { request } from "./client";
export const predictionApi = { predict: (model, payload) => request(`/api/v1/predictions/${model}`, { method: "POST", body: JSON.stringify(payload) }) };

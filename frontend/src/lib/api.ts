import { apiBaseUrl } from "./env";

export interface HealthResponse {
  status: "ok";
  app: string;
  environment: string;
}

export async function checkApiHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${apiBaseUrl}/health`, { signal });
  if (!response.ok) throw new Error(`API health check failed with status ${response.status}`);
  return response.json() as Promise<HealthResponse>;
}
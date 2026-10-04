import { apiBaseUrl } from "./env";
import { supabase } from "./supabase";

export interface HealthResponse {
  status: "ok";
  app: string;
  environment: string;
}

export interface ApiErrorBody {
  detail?: string;
  message?: string;
}

export class ApiError extends Error {
  status: number;
  body: ApiErrorBody | null;

  constructor(
    message: string,
    status: number,
    body: ApiErrorBody | null = null,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

export async function checkApiHealth(
  signal?: AbortSignal,
): Promise<HealthResponse> {
  const response = await fetch(`${apiBaseUrl}/health`, { signal });

  if (!response.ok) {
    throw new Error(
      `API health check failed with status ${response.status}`,
    );
  }

  return response.json() as Promise<HealthResponse>;
}

type ApiRequestOptions = Omit<RequestInit, "body"> & {
  body?: BodyInit | Record<string, unknown> | null;
};

export async function apiRequest<T>(
  path: string,
  options: ApiRequestOptions = {},
): Promise<T> {
  if (!supabase) {
    throw new ApiError(
      "Supabase authentication is not configured.",
      0,
    );
  }

  const {
    data: { session },
  } = await supabase.auth.getSession();

  if (!session?.access_token) {
    throw new ApiError("Authentication is required.", 401);
  }

  const headers = new Headers(options.headers);

  headers.set("Authorization", `Bearer ${session.access_token}`);

  let body: BodyInit | undefined;

  if (
    options.body !== undefined &&
    options.body !== null &&
    !(options.body instanceof FormData) &&
    typeof options.body !== "string" &&
    !(options.body instanceof Blob) &&
    !(options.body instanceof ArrayBuffer)
  ) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.body);
  } else {
    body = options.body ?? undefined;
  }

  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...options,
    headers,
    body,
  });

  if (response.status === 204) {
    return undefined as T;
  }

  const contentType = response.headers.get("content-type") ?? "";

  const responseBody = contentType.includes("application/json")
    ? ((await response.json()) as unknown)
    : null;

  if (!response.ok) {
    const errorBody =
      responseBody &&
      typeof responseBody === "object" &&
      responseBody !== null
        ? (responseBody as ApiErrorBody)
        : null;

    const message =
      errorBody?.detail ??
      errorBody?.message ??
      `API request failed with status ${response.status}`;

    throw new ApiError(message, response.status, errorBody);
  }

  return responseBody as T;
}
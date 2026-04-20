import { config } from "../config";

export interface ApiError extends Error {
  status: number;
  body?: unknown;
}

export async function apiFetch<T>(
  path: string,
  accessToken: string,
  init: RequestInit = {},
): Promise<T> {
  const url = path.startsWith("http") ? path : `${config.apiEndpoint}${path}`;
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${accessToken}`);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const resp = await fetch(url, { ...init, headers });

  const text = await resp.text();
  const isJson = resp.headers.get("content-type")?.includes("application/json");
  const body = text ? (isJson ? JSON.parse(text) : text) : undefined;

  if (!resp.ok) {
    const err: ApiError = Object.assign(new Error(`${resp.status} ${resp.statusText}`), {
      status: resp.status,
      body,
    });
    throw err;
  }

  return body as T;
}

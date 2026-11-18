// Thin typed client for the FastAPI backend. In dev, Vite proxies /api to the API server.
export const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api/v1";

const TOKEN_KEY = "ts_access_token";

export const tokenStore = {
  get: (): string | null => {
    try {
      return sessionStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  },
  set: (t: string) => {
    try {
      sessionStorage.setItem(TOKEN_KEY, t);
    } catch {
      /* storage unavailable */
    }
  },
  clear: () => {
    try {
      sessionStorage.removeItem(TOKEN_KEY);
    } catch {
      /* storage unavailable */
    }
  },
};

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

// FastAPI returns either a string or a list of pydantic validation errors.
function formatDetail(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((e: { loc?: unknown[]; msg?: string }) => {
        const field = e.loc?.[e.loc.length - 1];
        return field ? `${String(field)}: ${e.msg}` : (e.msg ?? "");
      })
      .join("; ");
  }
  return null;
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = tokenStore.get();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = formatDetail(body.detail) ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export interface Health {
  status: string;
  version: string;
}
export interface Readiness {
  status: string;
  checks: Record<string, string>;
}
export interface ComputeInfo {
  platform: string;
  machine: string;
  python_version: string;
  torch_version: string | null;
  mps_available: boolean | null;
  cuda_available: boolean | null;
  recommended_device: string;
}
export interface User {
  id: string;
  email: string;
  full_name: string | null;
  is_active: boolean;
  is_admin: boolean;
  created_at: string;
}
export interface Token {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export const api = {
  health: () => apiFetch<Health>("/health"),
  ready: () => apiFetch<Readiness>("/health/ready"),
  compute: () => apiFetch<ComputeInfo>("/system/compute"),
  me: () => apiFetch<User>("/auth/me"),
  register: (email: string, password: string, full_name?: string) =>
    apiFetch<User>("/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, full_name }),
    }),
  login: (email: string, password: string) =>
    apiFetch<Token>("/auth/login", {
      method: "POST",
      body: new URLSearchParams({ username: email, password }),
    }),
};

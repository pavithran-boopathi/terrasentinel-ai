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

// ---------------------------------------------------------------- Phase 5–7 resources

export interface ModelVersion {
  id: string;
  name: string;
  version: string;
  task: string;
  arch: string;
  sha256: string;
  size_bytes: number;
  threshold: number;
  metrics: Record<string, unknown>;
  training: Record<string, unknown>;
  is_default: boolean;
  created_at: string;
}

export interface SatImage {
  id: string;
  filename: string;
  sha256: string;
  size_bytes: number;
  sensor: string;
  width: number;
  height: number;
  crs: string;
  footprint: GeoJSON.Polygon;
  bbox: [number, number, number, number];
  created_at: string;
}

export interface Exposure {
  crs: string;
  flood_area_km2: number;
  label: string;
  params: { min_building_overlap: number };
  buildings?: { total: number; potentially_exposed: number; exposed_footprint_m2: number };
  roads?: { total_km: number; flooded_km: number; segments_exposed: number };
}

export interface AnalysisSummary {
  model: string;
  threshold: number;
  device: string;
  flooded_area_km2: number;
  valid_area_km2: number;
  flooded_fraction_of_valid: number | null;
  polygons: number;
  seconds: number;
  pixels_valid: number;
  pixels_flooded: number;
  exposure?: Exposure;
}

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export interface Analysis {
  id: string;
  image_id: string;
  image_filename: string | null;
  model_name: string | null;
  model_version_id: string;
  status: JobStatus;
  progress: number;
  attempts: number;
  error: string | null;
  summary: AnalysisSummary | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  files: string[];
}

export interface Monitoring {
  jobs_by_status: Record<string, number>;
  succeeded_last_24h: number;
  failed_last_24h: number;
  duration_seconds: { mean: number | null; p95: number | null };
  per_model: { model: string; succeeded_jobs: number; mean_inference_seconds: number | null }[];
}

export interface Stat {
  n: number;
  mean: number | null;
  std: number;
  min: number | null;
  max: number | null;
}

export interface ModelGroup {
  arch: string;
  split_scheme: string;
  seeds: number[];
  params: number;
  runs: string[];
  splits: Record<
    string,
    {
      iou: Stat;
      f1: Stat;
      precision: Stat;
      recall: Stat;
      iou_val_tuned_threshold?: Stat;
      per_event_iou: Record<string, Stat>;
    }
  >;
}

export interface BaselineRow {
  method: string;
  split_scheme: string;
  split: string;
  iou: number;
  f1: number;
  precision: number;
  recall: number;
  iou_ci95?: [number, number];
}

export interface Experiments {
  generated: string;
  note: string;
  sen1floods11: { models: ModelGroup[]; baselines: BaselineRow[] };
  latency: { device: string; rows: { arch: string; batch: number; ms_per_chip: number; params: number }[] } | null;
  spacenet8: {
    run: string;
    best_epoch: number;
    test: {
      pixel_iou: Record<string, number>;
      mean_iou_fg: number;
      object: {
        buildings_total: number;
        building_detection_rate: number | null;
        flooded_building: { precision: number; recall: number; f1: number };
        road_centerline_detection_rate: number | null;
        flooded_road_length: { precision: number; recall: number; f1: number };
      };
    };
    baselines_test?: Record<string, { flooded_building: { f1: number }; flooded_road_length: { f1: number } }>;
  }[];
}

function authHeaders(): HeadersInit {
  const t = tokenStore.get();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

/** Fetch an authenticated binary resource and trigger a browser download. */
export async function downloadFile(path: string, filename: string): Promise<void> {
  const res = await fetch(`${API_BASE}${path}`, { headers: authHeaders() });
  if (!res.ok) throw new ApiError(res.status, res.statusText);
  const url = URL.createObjectURL(await res.blob());
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  a.click();
  URL.revokeObjectURL(url);
}

/** Absolute URL for map image sources (auth header is added by the map's transformRequest). */
export const apiUrl = (path: string) => new URL(`${API_BASE}${path}`, window.location.origin).href;

export const resources = {
  models: () => apiFetch<ModelVersion[]>("/models"),
  images: () => apiFetch<SatImage[]>("/images"),
  uploadImage: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return apiFetch<SatImage>("/images", { method: "POST", body: fd });
  },
  analyses: () => apiFetch<Analysis[]>("/analyses"),
  analysis: (id: string) => apiFetch<Analysis>(`/analyses/${id}`),
  createAnalysis: (image_id: string, model_version_id?: string) =>
    apiFetch<Analysis>("/analyses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id, model_version_id }),
    }),
  exposure: (id: string, buildings: File | null, roads: File | null, minOverlap: number) => {
    const fd = new FormData();
    if (buildings) fd.append("buildings", buildings);
    if (roads) fd.append("roads", roads);
    return apiFetch<Exposure>(`/analyses/${id}/exposure?min_building_overlap=${minOverlap}`, {
      method: "POST",
      body: fd,
    });
  },
  floodGeojson: (id: string) =>
    apiFetch<GeoJSON.FeatureCollection>(`/analyses/${id}/files/flood_polygons.geojson`),
  exposedBuildings: (id: string) =>
    apiFetch<GeoJSON.FeatureCollection>(`/analyses/${id}/files/exposed_buildings.geojson`),
  monitoring: () => apiFetch<Monitoring>("/monitoring/summary"),
  experiments: () => apiFetch<Experiments>("/experiments"),
};

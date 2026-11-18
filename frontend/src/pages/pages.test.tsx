import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { tokenStore } from "../lib/api";
import { AuthProvider } from "../lib/auth";

const USER = { id: "u1", email: "ana@example.com", full_name: null, is_active: true, is_admin: true, created_at: "2026-11-18T00:00:00Z" };
const stat = (mean: number, std = 0.01) => ({ n: 3, mean, std, min: mean - std, max: mean + std });
const EXPERIMENTS = {
  generated: "2026-11-18",
  note: "test",
  sen1floods11: {
    models: [
      {
        arch: "segformer_b2", split_scheme: "official", seeds: [42, 1, 2], params: 27344257, runs: ["r1"],
        splits: { test: { iou: stat(0.6789), f1: stat(0.8), precision: stat(0.8), recall: stat(0.8), per_event_iou: { Ghana: stat(0.5) } } },
      },
    ],
    baselines: [{ method: "pixel_hgb", split_scheme: "official", split: "test", iou: 0.6389, f1: 0.78, precision: 0.88, recall: 0.7, iou_ci95: [0.5, 0.75] }],
  },
  latency: null,
  spacenet8: [],
};
const JOB = {
  id: "j1", image_id: "i1", image_filename: "bolivia.tif", model_name: "flood:1", model_version_id: "m1",
  status: "succeeded", progress: 1, attempts: 1, error: null, created_at: "2026-11-18T00:00:00Z", started_at: null, finished_at: null,
  files: ["flood_mask.tif", "probability.tif", "flood_polygons.geojson", "summary.json"],
  summary: { model: "flood:1", threshold: 0.5, device: "cpu", flooded_area_km2: 4.1756, valid_area_km2: 25.4, flooded_fraction_of_valid: 0.16, polygons: 59, seconds: 1.3, pixels_valid: 1, pixels_flooded: 1 },
};

function mockApi(routes: Record<string, unknown>) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const key = Object.keys(routes).sort((a, b) => b.length - a.length).find((k) => url.endsWith(k));
      return key ? new Response(JSON.stringify(routes[key]), { status: 200 }) : new Response("{}", { status: 404 });
    }),
  );
}

function renderAt(path: string) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <AuthProvider>
        <MemoryRouter initialEntries={[path]}>
          <App />
        </MemoryRouter>
      </AuthProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => tokenStore.set("tok"));
afterEach(() => {
  vi.unstubAllGlobals();
  tokenStore.clear();
});

describe("signed-in pages", () => {
  it("Model Laboratory shows metrics from the experiments endpoint, not hardcoded values", async () => {
    mockApi({ "/auth/me": USER, "/models": [], "/experiments": EXPERIMENTS });
    renderAt("/models");
    expect(await screen.findByText(/0\.679 ± 0\.01/)).toBeInTheDocument();
    expect(screen.getByText("pixel_hgb")).toBeInTheDocument();
    expect(screen.getByText("SegFormer-B2")).toBeInTheDocument();
  });

  it("Reports offers only downloads that exist for an analysis", async () => {
    mockApi({ "/auth/me": USER, "/analyses": [JOB] });
    renderAt("/reports");
    expect(await screen.findByText("bolivia.tif")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "PDF report" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Mask GeoTIFF" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Exposed buildings" })).not.toBeInTheDocument();
  });

  it("AI Analysis explains when no model is registered", async () => {
    mockApi({ "/auth/me": USER, "/models": [], "/images": [] });
    renderAt("/analysis");
    expect(await screen.findByText(/No models registered/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start inference" })).toBeDisabled();
  });

  it("Dashboard lists recent analyses with flood area from the API", async () => {
    mockApi({
      "/auth/me": USER, "/analyses": [JOB], "/models": [],
      "/monitoring/summary": { jobs_by_status: { succeeded: 1 }, succeeded_last_24h: 1, failed_last_24h: 0, duration_seconds: { mean: 2, p95: 2 }, per_model: [] },
    });
    renderAt("/");
    expect(await screen.findByText("bolivia.tif")).toBeInTheDocument();
    expect(screen.getByText("4.176")).toBeInTheDocument();
  });
});

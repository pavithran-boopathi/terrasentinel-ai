import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { AuthProvider } from "./lib/auth";

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

function mockFetch(routes: Record<string, unknown>) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const key = Object.keys(routes).find((k) => url.endsWith(k));
      if (!key) return new Response("{}", { status: 404 });
      return new Response(JSON.stringify(routes[key]), { status: 200 });
    }),
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("App", () => {
  it("renders navigation for all main pages", () => {
    mockFetch({});
    renderAt("/");
    for (const label of ["Dashboard", "Satellite Explorer", "AI Analysis", "Model Laboratory"]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
  });

  it("shows live API status and compute device from the backend", async () => {
    mockFetch({
      "/health": { status: "ok", version: "0.1.0" },
      "/health/ready": { status: "ok", checks: { database: "ok" } },
      "/system/compute": {
        platform: "Darwin",
        machine: "arm64",
        python_version: "3.12",
        torch_version: "2.15.0",
        mps_available: true,
        cuda_available: false,
        recommended_device: "mps",
      },
    });
    renderAt("/");
    expect(await screen.findByText(/Online · v0.1.0/)).toBeInTheDocument();
    expect(await screen.findByText("mps")).toBeInTheDocument();
  });

  it("reports an unreachable backend honestly", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("", { status: 502 })));
    renderAt("/");
    expect(await screen.findByText(/Unreachable/)).toBeInTheDocument();
  });

  it("redirects protected pages to login when signed out", () => {
    mockFetch({});
    renderAt("/analysis");
    expect(screen.getByRole("heading", { name: "Sign in" })).toBeInTheDocument();
  });
});

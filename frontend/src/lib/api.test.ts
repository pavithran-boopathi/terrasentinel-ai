import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, apiFetch, tokenStore } from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
  tokenStore.clear();
});

describe("apiFetch", () => {
  it("formats FastAPI validation errors into readable messages", async () => {
    const detail = [{ loc: ["body", "password"], msg: "String should have at least 10 characters" }];
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail }), { status: 422 })));
    await expect(apiFetch("/x")).rejects.toThrow("password: String should have at least 10 characters");
  });

  it("exposes HTTP status on errors", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "nope" }), { status: 401 })));
    const err = (await apiFetch("/x").catch((e: unknown) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(401);
    expect(err.message).toBe("nope");
  });

  it("attaches the bearer token when signed in", async () => {
    tokenStore.set("tok123");
    const f = vi.fn(async () => new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", f);
    await apiFetch("/x");
    const headers = (f.mock.calls[0] as unknown as [string, RequestInit])[1].headers as Headers;
    expect(headers.get("Authorization")).toBe("Bearer tok123");
  });
});

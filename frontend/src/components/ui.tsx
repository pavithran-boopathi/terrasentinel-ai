import type { ReactNode } from "react";

export function PageHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <header className="border-b border-slate-800 px-8 py-6">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      {subtitle && <p className="mt-1 text-sm text-slate-400">{subtitle}</p>}
    </header>
  );
}

export function Card({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/50 p-5">
      <h2 className="mb-3 text-sm font-medium text-slate-300">{title}</h2>
      {children}
    </section>
  );
}

/** Honest placeholder for functionality scheduled in a later phase. Never shows fake data. */
export function NotYetBuilt({ phase, what }: { phase: number; what: string }) {
  return (
    <div className="rounded-lg border border-dashed border-slate-700 p-6 text-sm text-slate-400">
      <span className="mr-2 rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-300">
        Phase {phase}
      </span>
      {what} is not implemented yet. No results are shown until real models and data exist.
    </div>
  );
}

export function StatusDot({ ok }: { ok: boolean | null }) {
  const color = ok === null ? "bg-slate-500" : ok ? "bg-emerald-400" : "bg-rose-400";
  return <span className={`inline-block h-2 w-2 rounded-full ${color}`} aria-hidden />;
}

export function SignInPrompt({ what }: { what: string }) {
  return (
    <div className="rounded-lg border border-slate-800 p-6 text-sm text-slate-400">
      <a href="/login" className="text-sky-400 hover:underline">
        Sign in
      </a>{" "}
      to {what}.
    </div>
  );
}

export function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
      {hint && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const color: Record<string, string> = {
    succeeded: "bg-emerald-500/15 text-emerald-300",
    running: "bg-sky-500/15 text-sky-300",
    queued: "bg-slate-500/20 text-slate-300",
    failed: "bg-rose-500/15 text-rose-300",
  };
  return <span className={`rounded px-2 py-0.5 text-xs ${color[status] ?? "bg-slate-700"}`}>{status}</span>;
}

export const fmt = (v: number | null | undefined, digits = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toLocaleString(undefined, { maximumFractionDigits: digits });

export const btn =
  "rounded-md bg-sky-600 px-3 py-1.5 text-sm font-medium hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50";
export const btnGhost = "rounded-md border border-slate-700 px-3 py-1.5 text-sm hover:bg-slate-800";
export const select =
  "rounded-md border border-slate-700 bg-slate-900 px-2 py-1.5 text-sm outline-none focus:border-sky-500";

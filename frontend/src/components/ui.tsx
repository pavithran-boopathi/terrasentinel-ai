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

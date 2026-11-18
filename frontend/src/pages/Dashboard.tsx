import { useQuery } from "@tanstack/react-query";
import { Card, NotYetBuilt, PageHeader, StatusDot } from "../components/ui";
import { api } from "../lib/api";

export default function Dashboard() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const ready = useQuery({ queryKey: ["ready"], queryFn: api.ready });
  const compute = useQuery({ queryKey: ["compute"], queryFn: api.compute });

  return (
    <>
      <PageHeader title="Dashboard" subtitle="Live system status from the TerraSentinel API" />
      <div className="grid gap-5 p-8 md:grid-cols-3">
        <Card title="API">
          {health.isPending ? (
            <p className="text-sm text-slate-400">Checking…</p>
          ) : health.isError ? (
            <p className="text-sm text-rose-300">
              <StatusDot ok={false} /> Unreachable — is the backend running on :8000?
            </p>
          ) : (
            <p className="text-sm">
              <StatusDot ok /> <span className="ml-1">Online · v{health.data.version}</span>
            </p>
          )}
        </Card>
        <Card title="Dependencies">
          {ready.data ? (
            <ul className="space-y-1 text-sm">
              {Object.entries(ready.data.checks).map(([k, v]) => (
                <li key={k} className="flex items-center gap-2">
                  <StatusDot ok={v === "ok"} /> <span className="capitalize">{k}</span>
                  <span className="text-slate-400">{v}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-slate-400">{ready.isError ? "Unavailable" : "Checking…"}</p>
          )}
        </Card>
        <Card title="Compute">
          {compute.data ? (
            <dl className="grid grid-cols-2 gap-y-1 text-sm">
              <dt className="text-slate-400">Device</dt>
              <dd className="font-mono uppercase">{compute.data.recommended_device}</dd>
              <dt className="text-slate-400">PyTorch</dt>
              <dd className="font-mono">{compute.data.torch_version ?? "not installed"}</dd>
              <dt className="text-slate-400">Machine</dt>
              <dd className="font-mono">{compute.data.machine}</dd>
            </dl>
          ) : (
            <p className="text-sm text-slate-400">
              {compute.isError ? "Unavailable" : "Checking…"}
            </p>
          )}
        </Card>
        <div className="md:col-span-3">
          <NotYetBuilt phase={3} what="Model evaluation metrics and recent assessments" />
        </div>
      </div>
    </>
  );
}
